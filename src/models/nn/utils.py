""" Utility wrappers around modules to let them handle Args and extra arguments """

import inspect
from functools import wraps
import torch
from torch import nn

'''
这个wrap_kwargs函数的目的是包装一个可调用对象f，使得当调用f时，任何未被f使用的关键字参数（kwargs）都会被返回。这样做可以使得函数更加灵活，允许传递额外的参数而不会影响函数的预期行为。下面是对这个函数的详细解释：

首先，函数定义了一个多行文档字符串，解释了wrap_kwargs的功能，并给出了几个使用示例。

使用inspect.signature获取可调用对象f的签名，这包含了f的所有参数信息。

通过检查f的签名中的参数，确定f是否已经定义了**kwargs参数。如果存在，has_kwargs将为True。

7-17. 如果f已经定义了**kwargs参数，那么定义一个新的函数f_kwargs，它使用@wraps装饰器来保留原函数f的元信息。f_kwargs接受任意位置参数*args和任意关键字参数**kwargs，然后调用原函数f。如果原函数返回的是一个元组，并且元组的最后一个元素是一个字典，那么这个字典会被提取出来，并与传入的**kwargs合并后返回。否则，返回原函数的返回值和一个空字典。

19-35. 如果f没有定义**kwargs参数，那么创建一个新的Parameter对象param_kwargs，表示**kwargs，并创建一个新的签名sig_kwargs，将f的所有原有参数加上新的**kwargs参数。

37-49. 定义一个新的函数f_kwargs，它使用新的签名sig_kwargs来绑定传入的参数。然后检查绑定后的参数字典中是否包含"kwargs"键，如果包含，则将其提取出来；否则，创建一个空字典。接着调用原函数f，并传入绑定后的参数。如果原函数返回的是一个元组，并且元组的最后一个元素是一个字典，那么这个字典会被提取出来，并与传入的**kwargs合并后返回。否则，返回原函数的返回值和剩余的关键字参数字典。

最后，返回新定义的函数f_kwargs。
总的来说，wrap_kwargs函数通过检查原函数的参数签名，并相应地创建一个新的包装函数，使得额外的关键字参数可以被返回，而不是被忽略。这对于创建灵活的函数包装器非常有用，尤其是在处理那些可能需要不同参数集的函数时。
'''
def wrap_kwargs(f):
    """
    Given a callable f that can consume some named arguments,
    wrap it with a kwargs that passes back any unused args

    EXAMPLES
    --------

    Basic usage:
    def foo(x, y=None):
        return x

    wrap_kwargs(foo)(0, y=1, z=2) == (0, {'z': 2})

    --------

    The wrapped function can return its own argument dictionary,
    which gets merged with the new kwargs.
    def foo(x, y=None):
        return x, {}
    wrap_kwargs(foo)(0, y=1, z=2) == (0, {'z': 2})

    def foo(x, y=None):
        return x, {"y": y, "z": None}
    wrap_kwargs(foo)(0, y=1, z=2) == (0, {'y': 1, 'z': 2})

    --------

    The wrapped function can have its own kwargs parameter:
    def foo(x, y=None, **kw_args):
        return x, {}
    wrap_kwargs(foo)(0, y=1, z=2) == (0, {})

    --------

    Partial functions and modules work automatically:
    class Module:
        def forward(self, x, y=0):
            return x, {"y": y+1}

    m = Module()

    wrap_kwargs(m.forward)(0, y=1, z=2) == (0, {'y': 2, 'z': 2})

    """
    sig = inspect.signature(f)
    # Check if f already has kwargs
    has_kwargs = any([
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    ])
    if has_kwargs:
        @wraps(f)
        def f_kwargs(*args, **kwargs):
            y = f(*args, **kwargs)
            if isinstance(y, tuple) and isinstance(y[-1], dict):
                return y
            else:
                return y, {}
    else:
        param_kwargs = inspect.Parameter("kwargs", kind=inspect.Parameter.VAR_KEYWORD)
        sig_kwargs = inspect.Signature(parameters=list(sig.parameters.values())+[param_kwargs])
        @wraps(f)
        def f_kwargs(*args, **kwargs):
            bound = sig_kwargs.bind(*args, **kwargs)
            if "kwargs" in bound.arguments:
                kwargs = bound.arguments.pop("kwargs")
            else:
                kwargs = {}
            y = f(**bound.arguments)
            if isinstance(y, tuple) and isinstance(y[-1], dict):
                return *y[:-1], {**y[-1], **kwargs}
            else:
                return y, kwargs
    return f_kwargs

'''

这段代码定义了一个名为 discard_kwargs 的函数，它的主要作用是包装另一个函数 f，以便在调用时忽略所有额外的关键字参数（kwargs）。下面是对这个函数的详细解读：

def discard_kwargs(f): 定义了一个名为 discard_kwargs 的函数，它接受一个参数 f，这个参数预期是一个可调用对象（例如函数或方法）。

if f is None: return None 检查传入的参数 f 是否为 None。如果是，函数直接返回 None。这是一个简单的保护措施，以避免对 None 类型进行进一步的操作。

5-13. f_kwargs = wrap_kwargs(f) 调用之前定义的 wrap_kwargs 函数来包装 f 函数。这样做的目的是创建一个新的函数，该函数能够捕获并返回任何未被 f 使用的关键字参数。

15-21. @wraps(f) 是一个装饰器，用于保留原函数 f 的元信息（如函数名、文档字符串等）。这是通过 functools.wraps 实现的，确保包装后的函数 f_ 在某些方面（如调试）表现得像原函数 f。

23-29. 定义了一个名为 f_ 的内部函数，它接受任意数量的位置参数 *args 和任意数量的关键字参数 **kwargs。这个函数将调用 f_kwargs 函数，并传入所有接收到的参数。然后，它返回 f_kwargs 返回值的第一个元素（[0] 表示索引为 0 的元素，即第一个元素）。

return f_ 返回定义好的 f_ 函数。这个函数现在可以替代原函数 f 被调用，它会忽略所有额外的关键字参数，只返回原函数的输出。
总的来说，discard_kwargs 函数提供了一种机制，使得在调用一个函数时，可以传递任意数量的额外关键字参数，而不用担心这些参数会影响函数的行为。这是因为所有未被原函数使用的关键字参数都会被 wrap_kwargs 捕获并返回，然后 discard_kwargs 通过返回第一个返回值来忽略这些参数。这在创建需要灵活处理参数的函数包装器时非常有用。
'''
def discard_kwargs(f):
    if f is None: return None
    f_kwargs = wrap_kwargs(f)
    @wraps(f)
    def f_(*args, **kwargs):
        return f_kwargs(*args, **kwargs)[0]
    return f_

def PassthroughSequential(*modules):
    """Special Sequential module that chains kwargs.

    Semantics are the same as nn.Sequential, with extra convenience features:
    - Discard None modules
    - Flatten inner Sequential modules
    - In case with 0 or 1 Module, rename the class for ease of inspection
    """

    '''
    定义了一个名为 flatten 的内部函数，它递归地展平 nn.Sequential 模块。如果传入的模块是 nn.Sequential 的实例，flatten 函数会递归地展平其中的所有子模块，并将它们收集到一个列表中。如果传入的模块不是 nn.Sequential，它将直接返回一个包含该模块的列表。
    '''
    def flatten(module):
        if isinstance(module, nn.Sequential):
            return sum([flatten(m) for m in module], [])
        else:
            return [module]

    modules = flatten(nn.Sequential(*modules))
    modules = [module for module in modules if module if not None]
    '''
    定义了一个名为 Sequential 的内部类，它继承自 PyTorch 的 nn.Sequential 类。这个类重写了 forward 和 step 方法，以便在调用时处理关键字参数。
    '''
    class Sequential(nn.Sequential):
        '''
        forward 方法定义了数据通过 Sequential 模块的前向传播行为。它遍历所有的层，并使用 wrap_kwargs 函数调用每一层的 forward 方法，同时传递输入数据 x 和所有关键字参数 kwargs。然后，它返回处理后的数据和剩余的关键字参数。
        '''
        def forward(self, x, **kwargs):
            for layer in self:
                x, kwargs = wrap_kwargs(layer.forward)(x, **kwargs)
            return x, kwargs
        
        '''
         step 方法类似于 forward 方法，但它使用 step 方法（如果存在）来处理每一层，否则使用 forward 方法。这可以用于循环神经网络（RNN）中的步进操作。
         '''
        def step(self, x, **kwargs):
            for layer in self:
                fn = getattr(layer, "step", layer.forward)
                x, kwargs = wrap_kwargs(fn)(x, **kwargs)
            return x, kwargs

    '''
    根据传入的模块数量，动态地更改 Sequential 类的名称。如果没有任何模块，将其名称设置为 Identity，表示这个 Sequential 实际上不执行任何操作。如果只有一个模块，将其名称设置为该模块的类型名称，这样可以更容易地通过类名识别它。
    '''
    if len(modules) == 0:
        Sequential.__name__ = "Identity"
    elif len(modules) == 1:
        Sequential.__name__ = type(modules[0]).__name__
    return Sequential(*modules)
