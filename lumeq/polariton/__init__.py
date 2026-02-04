from pyscf import lib

from .qed_ks import polariton_cs
from .qed_ks import polariton_ns

from .qed_ks_grad import Gradients, Gradients2
from .qed_ks_hess import Hessian

polariton_cs.Gradients = lib.class_as_method(Gradients)
polariton_ns.Gradients = lib.class_as_method(Gradients2)

polariton_cs.Hessian = lib.class_as_method(Hessian)
