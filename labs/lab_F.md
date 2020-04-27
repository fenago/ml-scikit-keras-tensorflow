
Special Data Structures
========================

In this lab we will take a very quick look at the
data structures supported by TensorFlow, beyond regular float or integer
tensors. This includes strings, ragged tensors, sparse tensors, tensor
arrays, sets, and queues.



Strings
=======

Tensors can hold byte strings, which is useful in
particular for natural language processing (see lab 16)

``` {data-type="programlisting" code-language="pycon"}
>>> tf.constant(b"hello world")
<tf.Tensor: id=149, shape=(), dtype=string, numpy=b'hello world'>
```

If you try to build a tensor with a Unicode string, TensorFlow
automatically encodes it to UTF-8:

``` {data-type="programlisting" code-language="pycon"}
>>> tf.constant("café")
<tf.Tensor: id=138, shape=(), dtype=string, numpy=b'caf\xc3\xa9'>
```

It is also possible to create tensors representing Unicode strings. Just
create an array of 32-bit integers, each representing a single Unicode
code
point: ^[1]


``` {data-type="programlisting" code-language="pycon"}
>>> tf.constant([ord(c) for c in "café"])
<tf.Tensor: id=211, shape=(4,), dtype=int32,
            numpy=array([ 99,  97, 102, 233], dtype=int32)>
```


###### Note

In tensors of type tf.string, the string length is not part of the
tensor's shape. In other words, strings are considered as atomic values.
However, in a Unicode string tensor (i.e., an int32 tensor), the length
of the string *is* part of the tensor's shape.


The `tf.strings` package contains several functions to manipulate string
tensors, such as `length()` to count the number of bytes in a byte
string (or the number of code points if you set `unit="UTF8_CHAR"`),
`unicode_encode()` to convert a Unicode string tensor (i.e., int32
tensor) to a byte string tensor, and `unicode_decode()` to do the
reverse:

``` {data-type="programlisting" code-language="pycon"}
>>> b = tf.strings.unicode_encode(u, "UTF-8")
>>> tf.strings.length(b, unit="UTF8_CHAR")
<tf.Tensor: id=386, shape=(), dtype=int32, numpy=4>
>>> tf.strings.unicode_decode(b, "UTF-8")
<tf.Tensor: id=393, shape=(4,), dtype=int32,
            numpy=array([ 99,  97, 102, 233], dtype=int32)>
```

You can also manipulate tensors containing multiple strings:

``` {data-type="programlisting" code-language="pycon"}
>>> p = tf.constant(["Café", "Coffee", "caffè", "咖啡"])
>>> tf.strings.length(p, unit="UTF8_CHAR")
<tf.Tensor: id=299, shape=(4,), dtype=int32,
            numpy=array([4, 6, 5, 2], dtype=int32)>
>>> r = tf.strings.unicode_decode(p, "UTF8")
>>> r
tf.RaggedTensor(values=tf.Tensor(
[   67    97   102   233    67   111   102   102   101   101    99    97
   102   102   232 21654 21857], shape=(17,), dtype=int32),
   row_splits=tf.Tensor([ 0  4 10 15 17], shape=(5,), dtype=int64))
>>> print(r)
<tf.RaggedTensor [[67, 97, 102, 233], [67, 111, 102, 102, 101, 101],
                  [99, 97, 102, 102, 232], [21654, 21857]]>
```

Notice that the decoded strings are stored in a `RaggedTensor`. What is
that?




Ragged Tensors
==============

A ragged tensor is a special kind of tensor that
represents a list of arrays of different sizes. More generally, it is a
tensor with one or more *ragged dimensions*, meaning dimensions whose
slices may have different lengths. In the ragged tensor `r`, the second
dimension is a ragged dimension. In all ragged tensors, the first
dimension is always a regular dimension (also called a *uniform
dimension*).

All the elements of the ragged tensor `r` are regular tensors. For
example, let's look at the second element of the ragged tensor:

``` {data-type="programlisting" code-language="pycon"}
>>> print(r[1])
tf.Tensor([ 67 111 102 102 101 101], shape=(6,), dtype=int32)
```

The `tf.ragged` package contains several functions to create and
manipulate ragged tensors. Let's create a second ragged tensor using
`tf.ragged.constant()` and concatenate it with the first ragged tensor,
along axis 0:

``` {data-type="programlisting" code-language="pycon"}
>>> r2 = tf.ragged.constant([[65, 66], [], [67]])
>>> print(tf.concat([r, r2], axis=0))
<tf.RaggedTensor [[67, 97, 102, 233], [67, 111, 102, 102, 101, 101], [99, 97,
102, 102, 232], [21654, 21857], [65, 66], [], [67]]>
```

The result is not too surprising: the tensors in `r2` were appended
after the tensors in `r` along axis 0. But what if we concatenate `r`
and another ragged tensor along axis 1?

``` {data-type="programlisting" code-language="pycon"}
>>> r3 = tf.ragged.constant([[68, 69, 70], [71], [], [72, 73]])
>>> print(tf.concat([r, r3], axis=1))
<tf.RaggedTensor [[67, 97, 102, 233, 68, 69, 70], [67, 111, 102, 102, 101, 101,
71], [99, 97, 102, 102, 232], [21654, 21857, 72, 73]]>
```

This time, notice that the *i*^th^ tensor in `r` and the *i*^th^ tensor
in `r3` were concatenated. Now that's more unusual, since all of these
tensors can have different lengths.

If you call the `to_tensor()` method, it gets converted to a regular
tensor, padding shorter tensors with zeros to get tensors of equal
lengths (you can change the default value by setting the `default_value`
argument):

``` {data-type="programlisting" code-language="pycon"}
>>> r.to_tensor()
<tf.Tensor: id=1056, shape=(4, 6), dtype=int32, numpy=
array([[   67,    97,   102,   233,     0,     0],
       [   67,   111,   102,   102,   101,   101],
       [   99,    97,   102,   102,   232,     0],
       [21654, 21857,     0,     0,     0,     0]], dtype=int32)>
```

Many TF operations support ragged tensors. For the full list, see the
documentation of the `tf.RaggedTensor` class.




Sparse Tensors
==============

TensorFlow can also efficiently represent *sparse
tensors* (i.e., tensors containing mostly zeros). Just create a
`tf.SparseTensor`, specifying the indices and values of the nonzero
elements and the tensor's shape. The indices must be listed in "reading
order" (from left to right, and top to bottom). If you are unsure, just
use `tf.sparse.reorder()`. You can convert a sparse tensor to a dense
tensor (i.e., a regular tensor) using `tf.sparse.to_dense()`:

``` {data-type="programlisting" code-language="pycon"}
>>> s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],
                        values=[1., 2., 3.],
                        dense_shape=[3, 4])
>>> tf.sparse.to_dense(s)
<tf.Tensor: id=1074, shape=(3, 4), dtype=float32, numpy=
array([[0., 1., 0., 0.],
       [2., 0., 0., 0.],
       [0., 0., 0., 3.]], dtype=float32)>
```

Note that sparse tensors do not support as many operations as dense
tensors. For example, you can multiply a sparse tensor by any scalar
value, and you get a new sparse tensor, but you cannot add a scalar
value to a sparse tensor, as this would not return a sparse tensor:

``` {data-type="programlisting" code-language="pycon"}
>>> s * 3.14
<tensorflow.python.framework.sparse_tensor.SparseTensor at 0x13205d470>
>>> s + 42.0
[...] TypeError: unsupported operand type(s) for +: 'SparseTensor' and 'float'
```




Tensor Arrays
=============

A `tf.TensorArray` represents a list of tensors.
This can be handy in dynamic models containing loops, to accumulate
results and later compute some statistics. You can read or write tensors
at any location in the array:

``` {data-type="programlisting" code-language="python"}
array = tf.TensorArray(dtype=tf.float32, size=3)
array = array.write(0, tf.constant([1., 2.]))
array = array.write(1, tf.constant([3., 10.]))
array = array.write(2, tf.constant([5., 7.]))
tensor1 = array.read(1) # => returns (and pops!) tf.constant([3., 10.])
```

Notice that reading an item pops it from the array, replacing it with a
tensor of the same shape, full of zeros.


###### Warning

When you write to the array, you must assign the output back to the
array, as shown in this code example. If you don't, although your code
will work fine in eager mode, it will break in graph mode (these modes
were presented in
[Lab 12]


When creating a `TensorArray`, you must provide its `size`, except in
graph mode. Alternatively, you can leave the `size` unset and instead
set `dynamic_size=True`, but this will hinder performance, so if you
know the `size` in advance, you should set it. You must also specify the
`dtype`, and all elements must have the same shape as the first one
written to the array.

You can stack all the items into a regular tensor by calling the
`stack()` method:

``` {data-type="programlisting" code-language="pycon"}
>>> array.stack()
<tf.Tensor: id=2110875, shape=(3, 2), dtype=float32, numpy=
array([[1., 2.],
       [0., 0.],
       [5., 7.]], dtype=float32)>
```




Sets
====

TensorFlow supports sets of integers or strings
(but not floats). It represents them using regular tensors. For example,
the set `{1, 5, 9}` is just represented as the tensor `[[1, 5, 9]]`.
Note that the tensor must have at least two dimensions, and the sets
must be in the last dimension. For example, `[[1, 5, 9], [2, 5, 11]]` is
a tensor holding two independent sets: `{1, 5, 9}` and `{2, 5, 11}`. If
some sets are shorter than others, you must pad them with a padding
value (0 by default, but you can use any other value you prefer).

The `tf.sets` package contains several functions to manipulate sets. For
example, let's create two sets and compute their union (the result is a
sparse tensor, so we call `to_dense()` to display it):

``` {data-type="programlisting" code-language="pycon"}
>>> a = tf.constant([[1, 5, 9]])
>>> b = tf.constant([[5, 6, 9, 11]])
>>> u = tf.sets.union(a, b)
>>> u
<tensorflow.python.framework.sparse_tensor.SparseTensor at 0x132b60d30>
>>> tf.sparse.to_dense(u)
<tf.Tensor: [...] numpy=array([[ 1,  5,  6,  9, 11]], dtype=int32)>
```

You can also compute the union of multiple pairs of sets simultaneously:

``` {data-type="programlisting" code-language="pycon"}
>>> a = tf.constant([[1, 5, 9], [10, 0, 0]])
>>> b = tf.constant([[5, 6, 9, 11], [13, 0, 0, 0, 0]])
>>> u = tf.sets.union(a, b)
>>> tf.sparse.to_dense(u)
<tf.Tensor: [...] numpy=array([[ 1,  5,  6,  9, 11],
                               [ 0, 10, 13,  0,  0]], dtype=int32)>
```

If you prefer to use a different padding value, you must set
`default_value` when calling `to_dense()`:

``` {data-type="programlisting" code-language="pycon"}
>>> tf.sparse.to_dense(u, default_value=-1)
<tf.Tensor: [...] numpy=array([[ 1,  5,  6,  9, 11],
                               [ 0, 10, 13, -1, -1]], dtype=int32)>
```


###### Warning

The default `default_value` is 0, so when dealing with string sets, you
must set the `default_value` (e.g., to an empty string).


Other functions available in `tf.sets` include `difference()`,
`intersection()`, and `size()`, which are self-explanatory. If you want
to check whether or not a set contains some given values, you can
compute the intersection of that set and the values. If you want to add
some values to a set, you can compute the union of the set and the
values.




Queues
======

A queue is a data structure to which you can push
data records, and later pull them out. TensorFlow implements several
types of queues in the `tf.queue` package. They used to be very
important when implementing efficient data loading and preprocessing
pipelines, but the tf.data API has essentially rendered them useless
(except perhaps in some rare cases) because it is much simpler to use
and provides all the tools you need to build efficient pipelines. For
the sake of completeness, though, let's take a quick look at them.

The simplest kind of queue is the first-in, first-out (FIFO) queue. To
build it, you need to specify the maximum number of records it can
contain. Moreover, each record is a tuple of tensors, so you must
specify the type of each tensor, and optionally their shapes. For
example, the following code example creates a FIFO queue with maximum
three records, each containing a tuple with a 32-bit integer and a
string. Then it pushes two records to it, looks at the size (which is 2
at this point), and pulls a record out:

``` {data-type="programlisting" code-language="pycon"}
>>> q = tf.queue.FIFOQueue(3, [tf.int32, tf.string], shapes=[(), ()])
>>> q.enqueue([10, b"windy"])
>>> q.enqueue([15, b"sunny"])
>>> q.size()
<tf.Tensor: id=62, shape=(), dtype=int32, numpy=2>
>>> q.dequeue()
[<tf.Tensor: id=6, shape=(), dtype=int32, numpy=10>,
 <tf.Tensor: id=7, shape=(), dtype=string, numpy=b'windy'>]
```

It is also possible to enqueue and dequeue multiple records at once (the
latter requires specifying the shapes when creating the queue):

``` {data-type="programlisting" code-language="pycon"}
>>> q.enqueue_many([[13, 16], [b'cloudy', b'rainy']])
>>> q.dequeue_many(3)
[<tf.Tensor: [...] numpy=array([15, 13, 16], dtype=int32)>,
 <tf.Tensor: [...] numpy=array([b'sunny', b'cloudy', b'rainy'], dtype=object)>]
```

Other queue types include:

`PaddingFIFOQueue`

:   Same as `FIFOQueue`, but its `dequeue_many()` method supports
    dequeueing multiple records of different shapes. It automatically
    pads the shortest records to ensure all the records in the batch
    have the same shape.

`PriorityQueue`

:   A queue that dequeues records in a prioritized order. The priority
    must be a 64-bit integer included as the first element of each
    record. Surprisingly, records with a lower priority will be dequeued
    first. Records with the same priority will be dequeued in FIFO
    order.

`RandomShuffleQueue`

:   A queue whose records are dequeued in random order. This was useful
    to implement a shuffle buffer before tf.data existed.

If a queue is already full and you try to enqueue another record, the
`enqueue*()` method will freeze until a record is dequeued by another
thread. Similarly, if a queue is empty and you try to dequeue a record,
the `dequeue*()` method will freeze until records are pushed to the
queue by another thread. 



^[1]

If you are not familiar with Unicode code points, please check out
[*https://homl.info/unicode*](https://homl.info/unicode).
