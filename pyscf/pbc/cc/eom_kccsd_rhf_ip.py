#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Artem Pulkin, pyscf authors

from pyscf.lib import logger, linalg_helper, einsum
from pyscf.lib.parameters import LARGE_DENOM

from pyscf.pbc.lib.kpts_helper import nested_to_vector, vector_to_nested
from pyscf.pbc.mp.kmp2 import padding_k_idx

import numpy as np

import time


def vector_spec(cc):
    """Description of the IP vector."""
    return [(cc.nocc,), (cc.nkpts, cc.nkpts, cc.nocc, cc.nocc, cc.nmo - cc.nocc)]


def a2v(cc, t1, t2):
    """Ground state amplitudes to a vector."""
    return nested_to_vector((t1, t2))[0]


def v2a(cc, vec):
    """Ground state vector to apmplitudes."""
    return vector_to_nested(vec, vector_spec(cc))


def vector_size(cc):
    nocc = cc.nocc
    nvir = cc.nmo - nocc
    nkpts = cc.nkpts

    size = nocc + nkpts ** 2 * nocc ** 2 * nvir
    return size


def kernel(cc, nroots=1, koopmans=False, guess=None, partition=None,
           kptlist=None):
    '''Calculate (N-1)-electron charged excitations via IP-EOM-CCSD.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested per k-point
        koopmans : bool
            Calculate Koopmans'-like (quasiparticle) excitations only, targeting via
            overlap.
        guess : list of ndarray
            List of guess vectors to use for targeting via overlap.
        partition : bool or str
            Use a matrix-partitioning for the doubles-doubles block.
            Can be None, 'mp' (Moller-Plesset, i.e. orbital energies on the diagonal),
            or 'full' (full diagonal elements).
        kptlist : list
            List of k-point indices for which eigenvalues are requested.
    '''
    cput0 = (time.clock(), time.time())
    log = logger.Logger(cc.stdout, cc.verbose)
    nocc = cc.nocc
    nvir = cc.nmo - nocc
    nkpts = cc.nkpts
    if kptlist is None:
        kptlist = range(nkpts)
    size = vector_size(cc)
    for k, kshift in enumerate(kptlist):
        nfrozen = np.sum(mask_frozen(cc, np.zeros(size, dtype=int), kshift, const=1))
        nroots = min(nroots, size - nfrozen)
    if partition:
        partition = partition.lower()
        assert partition in ['mp', 'full']
    cc.ip_partition = partition
    evals = np.zeros((len(kptlist), nroots), np.float)
    evecs = np.zeros((len(kptlist), nroots, size), np.complex)

    for k, kshift in enumerate(kptlist):
        adiag = diag(cc, kshift)
        adiag = mask_frozen(cc, adiag, kshift, const=LARGE_DENOM)
        if partition == 'full':
            cc._ipccsd_diag_matrix2 = v2a(cc, adiag)[1]

        if guess is not None:
            guess_k = guess[k]
            assert len(guess_k) == nroots
            for g in guess_k:
                assert g.size == size
        else:
            guess_k = []
            if koopmans:
                # Get location of padded elements in occupied and virtual space
                nonzero_opadding = padding_k_idx(cc, kind="split")[0][kshift]

                for n in nonzero_opadding[::-1][:nroots]:
                    g = np.zeros(size)
                    g[n] = 1.0
                    g = mask_frozen(cc, g, kshift, const=0.0)
                    guess_k.append(g)
            else:
                idx = adiag.argsort()[:nroots]
                for i in idx:
                    g = np.zeros(size)
                    g[i] = 1.0
                    g = mask_frozen(cc, g, kshift, const=0.0)
                    guess_k.append(g)

        def precond(r, e0, x0):
            return r / (e0 - adiag + 1e-12)

        eig = linalg_helper.eig
        if guess is not None or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                idx = np.argmax(np.abs(np.dot(np.array(guess_k).conj(), np.array(x0).T)), axis=1)
                return linalg_helper._eigs_cmplx2real(w, v, idx)

            evals_k, evecs_k = eig(lambda _arg: matvec(cc, _arg, kshift), guess_k, precond, pick=pickeig,
                                   tol=cc.conv_tol, max_cycle=cc.max_cycle,
                                   max_space=cc.max_space, nroots=nroots, verbose=cc.verbose)
        else:
            evals_k, evecs_k = eig(lambda _arg: matvec(cc, _arg, kshift), guess_k, precond,
                                   tol=cc.conv_tol, max_cycle=cc.max_cycle,
                                   max_space=cc.max_space, nroots=nroots, verbose=cc.verbose)

        evals_k = evals_k.real
        evals[k] = evals_k
        evecs[k] = evecs_k

        if nroots == 1:
            evals_k, evecs_k = [evals_k], [evecs_k]

        for n, en, vn in zip(range(nroots), evals_k, evecs_k):
            r1, r2 = v2a(cc, vn)
            qp_weight = np.linalg.norm(r1) ** 2
            logger.info(cc, 'EOM root %d E = %.16g  qpwt = %0.6g',
                        n, en, qp_weight)
    log.timer('EOM-CCSD', *cput0)
    cc.eip = evals
    return cc.eip, evecs


def matvec(cc, vector, k):
    '''2ph operators are of the form s_{ij}^{ b}, i.e. 'jb' indices are coupled.'''
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    if not cc.imds.made_ip_imds:
        cc.imds.make_ip(cc.ip_partition)
    imds = cc.imds

    vector = mask_frozen(cc, vector, k, const=0.0)
    r1, r2 = v2a(cc, vector)

    t1, t2 = cc.t1, cc.t2
    nkpts = cc.nkpts
    kconserv = cc.khelper.kconserv

    # 1h-1h block
    Hr1 = -einsum('ki,k->i', imds.Loo[k], r1)
    # 1h-2h1p block
    for kl in range(nkpts):
        Hr1 += 2. * einsum('ld,ild->i', imds.Fov[kl], r2[k, kl])
        Hr1 += -einsum('ld,lid->i', imds.Fov[kl], r2[kl, k])
        for kk in range(nkpts):
            kd = kconserv[kk, k, kl]
            Hr1 += -2. * einsum('klid,kld->i', imds.Wooov[kk, kl, k], r2[kk, kl])
            Hr1 += einsum('lkid,kld->i', imds.Wooov[kl, kk, k], r2[kk, kl])

    Hr2 = np.zeros(r2.shape, dtype=np.common_type(imds.Wovoo[0, 0, 0], r1))
    # 2h1p-1h block
    for ki in range(nkpts):
        for kj in range(nkpts):
            kb = kconserv[ki, k, kj]
            Hr2[ki, kj] -= einsum('kbij,k->ijb', imds.Wovoo[k, kb, ki], r1)
    # 2h1p-2h1p block
    if cc.ip_partition == 'mp':
        nkpts, nocc, nvir = cc.t1.shape
        fock = cc.eris.fock
        foo = fock[:, :nocc, :nocc]
        fvv = fock[:, nocc:, nocc:]
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kconserv[ki, k, kj]
                Hr2[ki, kj] += einsum('bd,ijd->ijb', fvv[kb], r2[ki, kj])
                Hr2[ki, kj] -= einsum('li,ljb->ijb', foo[ki], r2[ki, kj])
                Hr2[ki, kj] -= einsum('lj,ilb->ijb', foo[kj], r2[ki, kj])
    elif cc.ip_partition == 'full':
        Hr2 += cc._ipccsd_diag_matrix2 * r2
    else:
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kconserv[ki, k, kj]
                Hr2[ki, kj] += einsum('bd,ijd->ijb', imds.Lvv[kb], r2[ki, kj])
                Hr2[ki, kj] -= einsum('li,ljb->ijb', imds.Loo[ki], r2[ki, kj])
                Hr2[ki, kj] -= einsum('lj,ilb->ijb', imds.Loo[kj], r2[ki, kj])
                for kl in range(nkpts):
                    kk = kconserv[ki, kl, kj]
                    Hr2[ki, kj] += einsum('klij,klb->ijb', imds.Woooo[kk, kl, ki], r2[kk, kl])
                    kd = kconserv[kl, kj, kb]
                    Hr2[ki, kj] += 2. * einsum('lbdj,ild->ijb', imds.Wovvo[kl, kb, kd], r2[ki, kl])
                    Hr2[ki, kj] += -einsum('lbdj,lid->ijb', imds.Wovvo[kl, kb, kd], r2[kl, ki])
                    Hr2[ki, kj] += -einsum('lbjd,ild->ijb', imds.Wovov[kl, kb, kj], r2[ki, kl])  # typo in Ref
                    kd = kconserv[kl, ki, kb]
                    Hr2[ki, kj] += -einsum('lbid,ljd->ijb', imds.Wovov[kl, kb, ki], r2[kl, kj])
        tmp = (2. * einsum('xyklcd,xykld->c', imds.Woovv[:, :, k], r2[:, :])
               - einsum('yxlkcd,xykld->c', imds.Woovv[:, :, k], r2[:, :]))
        Hr2[:, :] += -einsum('c,xyijcb->xyijb', tmp, t2[:, :, k])

    return mask_frozen(cc, a2v(cc, Hr1, Hr2), k, const=0.0)


def diag(cc, k):
    if not cc.imds.made_ip_imds:
        cc.imds.make_ip(cc.ip_partition)
    imds = cc.imds

    t1, t2 = cc.t1, cc.t2
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.khelper.kconserv

    Hr1 = -np.diag(imds.Loo[k])

    Hr2 = np.zeros((nkpts, nkpts, nocc, nocc, nvir), dtype=t1.dtype)
    if cc.ip_partition == 'mp':
        foo = cc.eris.fock[:, :nocc, :nocc]
        fvv = cc.eris.fock[:, nocc:, nocc:]
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kconserv[ki, k, kj]
                Hr2[ki, kj] = fvv[kb].diagonal()
                Hr2[ki, kj] -= foo[ki].diagonal()[:, None, None]
                Hr2[ki, kj] -= foo[kj].diagonal()[:, None]
    else:
        idx = np.arange(nocc)
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kconserv[ki, k, kj]
                Hr2[ki, kj] = imds.Lvv[kb].diagonal()
                Hr2[ki, kj] -= imds.Loo[ki].diagonal()[:, None, None]
                Hr2[ki, kj] -= imds.Loo[kj].diagonal()[:, None]

                if ki == kconserv[ki, kj, kj]:
                    Hr2[ki, kj] += np.einsum('ijij->ij', imds.Woooo[ki, kj, ki])[:, :, None]

                Hr2[ki, kj] -= np.einsum('jbjb->jb', imds.Wovov[kj, kb, kj])

                Wovvo = np.einsum('jbbj->jb', imds.Wovvo[kj, kb, kb])
                Hr2[ki, kj] += 2. * Wovvo
                if ki == kj:  # and i == j
                    Hr2[ki, ki, idx, idx] -= Wovvo

                Hr2[ki, kj] -= np.einsum('ibib->ib', imds.Wovov[ki, kb, ki])[:, None, :]

                kd = kconserv[kj, k, ki]
                Hr2[ki, kj] -= 2. * np.einsum('ijcb,jibc->ijb', t2[ki, kj, k], imds.Woovv[kj, ki, kd])
                Hr2[ki, kj] += np.einsum('ijcb,ijbc->ijb', t2[ki, kj, k], imds.Woovv[ki, kj, kd])

    return a2v(cc, Hr1, Hr2)


def mask_frozen(cc, vector, k, const=LARGE_DENOM):
    '''Replaces all frozen orbital indices of `vector` with the value `const`.'''
    r1, r2 = v2a(cc, vector)
    nkpts, nocc, nvir = cc.t1.shape
    kconserv = cc.khelper.kconserv

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind="split")

    new_r1 = const * np.ones_like(r1)
    new_r2 = const * np.ones_like(r2)

    new_r1[nonzero_opadding[k]] = r1[nonzero_opadding[k]]
    for ki in range(nkpts):
        for kj in range(nkpts):
            kb = kconserv[ki, k, kj]
            idx = np.ix_([ki], [kj], nonzero_opadding[ki], nonzero_opadding[kj], nonzero_vpadding[kb])
            new_r2[idx] = r2[idx]

    return a2v(cc, new_r1, new_r2)
