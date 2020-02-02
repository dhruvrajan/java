Tensorflow-Data (Java)
==

TensorFlow Data provides simple APIs for loading data of various formats, and preparing
datasets for use in training and using deep learning models.

TensorFlow Java's implementation simplifies the use of the C++ `data` ops, and provides
a simple API for configuring and iterating over datasets in both "graph" and "eager" mode.

Usage
--

A `Dataset` is an abstraction representing a sequence of elements. 
Each element in the sequence is a collection (`List`) of tensors (or, "components").

Creation
-
A dataset can be constructed from a list of constant tensors as in
the following example:

```java
// Declare dataset components as arrays.
// NOTE: All components in a dataset must share the first "batch" dimension.

int[][] m1 = new int[][]{
        {1, 2, 3, 4, 5},
        {2, 4, 6, 8, 10},
        {3, 6, 8, 12, 15},
        {4, 8, 12, 16, 20}
    };

int[][] m2 = new int[][]{
    {1}, {0}, {1}, {1}
};


Ops tf = // ... TensorFlow Ops Accessor (either graph or eager).

// Construct dataset with two components, batchSize=2.
Dataset dataset = Dataset.fromTensorSlices(tf,
    // List of array components
    Arrays.asList(tf.constant(m1), tf.constant(m2)),
    // List of each component's dtype
    Arrays.asList(TInt32.DTYPE, TInt32.DTYPE)
).batch(2);
```

Iteration
--

In eager mode, the dataset can be iterated through using a standard 
"for-each" loop, to receive the tensor values of each component:

```java
int BATCH_SIZE = 2;
for (List<Output<?>> components : dataset.batch(BATCH_SIZE)) {
      Tensor<TInt32> XBatch = components.get(0).tensor().expect(TInt32.DTYPE);
      Tensor<TInt32> yBatch = components.get(1).tensor().expect(TInt32.DTYPE);
      
      // ... use batch tensors
}
```

In graph mode, the dataset is iterated through using a while-loop, 
using the `OneShotIterator` abstraction.

```java
while (true) {
    try {
        IteratorGetNext getNext = tf.data.iteratorGetNext(anonymousIter.handle(), outputTypes, outputShapes);
        List<Output<?>> outputs = getNext.components();
        System.out.println("BATCH: ");
        printIntTensor(outputs.get(0).tensor());
        printIntTensor(outputs.get(1).tensor());
        System.out.println();
    } catch (IndexOutOfBoundsException e) {
        System.out.println("finished iterating.");
        break;
    }
}
```