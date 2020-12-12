# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)

"""
    Reduced Wong-Wang model with coupled excitatory-inhibitory populations.

    This model describes the functional dynamics of a local brain region, composed by excitatory–inhibitory subnetworks (e–i networks). The excitatory synaptic currents are mediated by NMDA receptors and the inhibitory currents are mediated by GABA-A receptors. The dynamics of the model describes the time evolution of the mean synaptic activity of each local region by following synaptic equations:
    
    Equations taken from [Deco_2014], page 7889
    .. math::
        I_{ek}                    &=   W_eI_o + w_p\,J_N\,S_{ek} - J_k\,S_{ik} + I_ext + GJ_N \sum_{j}u_{kj}S_{ek},\\
        r_{ek} = H_{e}(I_{ek})    &=  \dfrac{a_eI_{ek}- b_e}{1 - \exp(-d_e(a_eI_{ek} -b_e))},\\
        \dot{S}_{ek} &= -\dfrac{S_{ek}}{\tau_e} + (1 - S_{ek}) \, \gamma_e r_{ek}) \\
        
        I_{ik}                    &=   W_iI_o + J_N \, S_{ek} - S_{ik} +  \lambdaGJ_N \sum_{j}u_{kj}S_{ej},\\
        r_{ik} = H_{i}(I_{ik})    &=  \dfrac{a_iI_{ik} - b_i}{1 - \exp(-d_i(a_iI_{ik} -b_i))},\\
        \dot{S}_{ik} &= -\dfrac{S_{ik}}{\tau_i} + \gamma_i r_{ik}
        
        
    .. [Deco_2014] Deco Gustavo, Ponce Alvarez Adrian, Patric Hagmann, Gian Luca Romani, Dante Mantini, and Maurizio Corbetta. *How Local Excitation–Inhibition Ratio Impacts the Whole Brain Dynamics*. The Journal of Neuroscience 34(23), 7886 –7898, 2014.

    .. moduleauthor:: Julie Courtiol <courtiol.julie@gmail.com>

"""

#from .base import ModelNumbaDfun, LOG, numpy, basic, arrays
#from numba import guvectorize, float64

import numpy
from tvb.simulator.models.base import ModelNumbaDfun
from tvb.simulator.common import get_logger
import tvb.basic.traits.types_basic as basic
import tvb.datatypes.arrays as arrays
from numba import guvectorize, float64

@guvectorize([(float64[:],)*22], '(n),(m)' + ',()'*19 + '->(n)', nopython=True)
def _numba_dfun(S, c, ae, be, de, ge, te, wp, we, jn, ai, bi, di, gi, ti, wi, j, g, l, io, iext, dS):
    "Gufunction for Reduced Wong-Wang Excitatory-Inhibitory model equations."

    cc = g[0]*jn[0]*c[0]

    ie = we[0]*io[0] + wp[0]*jn[0]*S[0] - j[0]*S[1] + cc + iext[0]
    re = (ae[0]*ie - be[0]) / (1 - numpy.exp(-de[0]*(ae[0]*ie - be[0])))
    dS[0] = - (S[0] / te[0]) + (1.0 - S[0]) * ge[0] * re

    ii = wi[0]*io[0] + jn[0]*S[0] - S[1] + l[0]*cc
    ri = (ai[0]*ii - bi[0]) / (1 - numpy.exp(-di[0]*(ai[0]*ii - bi[0])))
    dS[1] = - (S[1] / ti[0]) + gi[0] * ri

class ReducedWongWangExcInh(ModelNumbaDfun):
    """
    """
    # Define traited attributes for this model, these represent possible kwargs.
    a_e = arrays.FloatArray(
        label=":math:`a_e`",
        default=numpy.array([310.0, ]),
        range=basic.Range(lo=0.0, hi=500.0, step=1.0),
        doc="[n/C]. Excitatory population input gain parameter.")

    b_e = arrays.FloatArray(
        label=":math:`b_e`",
        default=numpy.array([125.0, ]),
        range=basic.Range(lo=0.0, hi=200.0, step=1.0),
        doc="[Hz]. Excitatory population input shift parameter.")

    d_e = arrays.FloatArray(
        label=":math:`d_e`",
        default=numpy.array([0.160, ]),
        range=basic.Range(lo=0.0, hi=0.2, step=0.001),
        doc="""[s]. Excitatory population input scaling parameter.""")

    gamma_e = arrays.FloatArray(
        label=r":math:`\gamma_e`",
        default=numpy.array([0.641/1000, ]),
        range=basic.Range(lo=0.0, hi=1.0/1000, step=0.001/1000),
        doc="""Excitatory population kinetic parameter, the factor 1000 is for expressing everything in ms.""")

    tau_e = arrays.FloatArray(
        label=r":math:`\tau_e`",
        default=numpy.array([100.0, ]),
        range=basic.Range(lo=50.0, hi=150.0, step=1.0),
        doc="""[ms]. Excitatory population NMDA decay time constant.""")

    w_p = arrays.FloatArray(
        label=r":math:`w_p`",
        default=numpy.array([1.4, ]),
        range=basic.Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Excitatory population synaptic weight of the recurrence self-excitation.""")

    J_N = arrays.FloatArray(
        label=r":math:`J_{N}`",
        default=numpy.array([0.15, ]),
        range=basic.Range(lo=0.001, hi=0.5, step=0.001),
        doc="""[nA] Excitatory NMDA synaptic coupling.""")

    W_e = arrays.FloatArray(
        label=r":math:`W_e`",
        default=numpy.array([1.0, ]),
        range=basic.Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Excitatory population external input scaling weight.""")

    a_i = arrays.FloatArray(
        label=":math:`a_i`",
        default=numpy.array([615.0, ]),
        range=basic.Range(lo=0.0, hi=1000.0, step=1.0),
        doc="[n/C]. Inhibitory population input gain parameter.")

    b_i = arrays.FloatArray(
        label=":math:`b_i`",
        default=numpy.array([177.0, ]),
        range=basic.Range(lo=0.0, hi=200.0, step=1.0),
        doc="[Hz]. Inhibitory population input shift parameter.")

    d_i = arrays.FloatArray(
        label=":math:`d_i`",
        default=numpy.array([0.087, ]),
        range=basic.Range(lo=0.0, hi=0.2, step=0.001),
        doc="""[s]. Inhibitory population input scaling parameter.""")

    gamma_i = arrays.FloatArray(
        label=r":math:`\gamma_i`",
        default=numpy.array([1.0/1000, ]),
        range=basic.Range(lo=0.0, hi=2.0/1000, step=0.01/1000),
        doc="""Inhibitory population kinetic parameter, the factor 1000 is for expressing everything in ms.""")

    tau_i = arrays.FloatArray(
        label=r":math:`\tau_i`",
        default=numpy.array([10.0, ]),
        range=basic.Range(lo=50.0, hi=150.0, step=1.0),
        doc="""[ms]. Inhibitory population GABA decay time constant.""")

    J = arrays.FloatArray(
        label=r":math:`J`",
        default=numpy.array([1.0, ]),
        range=basic.Range(lo=0.001, hi=2.0, step=0.001),
        doc="""[nA] Local feedback inhibitory synaptic coupling. By default, in the no-FIC (feedback inhibitory control) case. Otherwise, it is adjusted independently by the algorithm described as in Deco_2014.""")

    W_i = arrays.FloatArray(
        label=r":math:`W_i`",
        default=numpy.array([0.7, ]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Inhibitory population external input scaling weight.""")

    I_o = arrays.FloatArray(
        label=":math:`I_{o}`",
        default=numpy.array([0.382, ]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.001),
        doc="""[nA]. Effective external input.""")

    G = arrays.FloatArray(
        label=":math:`G`",
        default=numpy.array([1.0, ]),
        range=basic.Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Global coupling scaling.""")

    lamda = arrays.FloatArray(
        label=":math:`\lambda`",
        default=numpy.array([0.0, ]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Inhibitory global coupling scaling. By default, in the no-FFI (feedforward inhibition) case. Otherwise, it is set to 1.""")
        
    I_ext = arrays.FloatArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.0, ]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""External stimulation input for simulating task-evoked activity. By default, in resting-state activity. Otherwise, it is set to 0.02.""")

    state_variable_range = basic.Dict(
        label="State-variable ranges [lo, hi]",
        default={
            "S_e": numpy.array([0.0, 1.0]),
            "S_i": numpy.array([0.0, 1.0])
        },
        doc="Population mean synaptic gating variables.")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    #state_variable_boundaries = basic.Enumerate(
        #label="State-variable boundaries [lo, hi]",
        #default={"S_e": numpy.array([0.0, 1.0]), "S_i": numpy.array([0.0, 1.0])},
        #doc="""The values for each state-variable should be set to encompass the boundaries of the dynamic range of that state-variable. Set None for one-sided boundaries.""")

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=['S_e', 'S_i'],
        default=['S_e', 'S_i'],
        select_multiple=True,
        doc="Default state-variables to be monitored.")

    state_variables = ['S_e', 'S_i']
    _nvar = 2
    cvar = numpy.array([0], dtype=numpy.int32)

    def configure(self):
        """  """
        super(ReducedWongWangExcInh, self).configure()
        self.update_derived_parameters()

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):
        """
            Computes the derivatives of the state-variables of the ReducedWongWangExcInh model with respect to time.
        """
        S = state_variables[:, :]

        c_0 = coupling[0, :]
        lc_0 = local_coupling * S[0] # if applicable

        coupling = self.G * self.J_N * (c_0 + lc_0)

        I_e = self.W_e * self.I_o + self.w_p * self.J_N * S[0] - self.J * S[1] + coupling + self.I_ext
        r_e = (self.a_e * I_e - self.b_e) / (1 - numpy.exp(-self.d_e * (self.a_e * I_e - self.b_e)))
        dS_e = - (S[0] / self.tau_e) + (1 - S[0]) * self.gamma_e * r_e

        I_i = self.W_i * self.I_o + self.J_N * S[0] - S[1] + self.lamda * coupling
        r_i = (self.a_i * I_i - self.b_i) / (1 - numpy.exp(-self.d_i * (self.a_i * I_i - self.b_i)))
        dS_i = - (S[1] / self.tau_i) + self.gamma_i * r_i

        derivative = numpy.array([dS_e, dS_i])
        return derivative

    def dfun(self, x, c, local_coupling=0.0, **kwargs):
        """The dfun using numba for speed."""
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T + local_coupling * x[0]
        deriv = _numba_dfun(x_, c_,
                            self.a_e, self.b_e, self.d_e, self.gamma_e, self.tau_e,
                            self.w_p, self.W_e, self.J_N,
                            self.a_i, self.b_i, self.d_i, self.gamma_i, self.tau_i,
                            self.W_i, self.J,
                            self.G, self.lamda, self.I_o, self.I_ext)
        return deriv.T[..., numpy.newaxis]
