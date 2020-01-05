//package org.tensorflow.keras.utils;
//
//import java.lang.Math;
//import org.tensorflow.Shape;
//import org.tensorflow.Tensors;
//
//import static org.tensorflow.keras.utils.Keras.*;
//
///**
// * Wraps a org.tensorflow.Shape to provide convenient
// * features for verifying and modifying tensor shapes.
// */
//public class TensorShape {
//  private long[] dims;
//
//  public TensorShape(long head, long... tail) {
//    this.dims = new long[tail.length + 1];
//    this.dims[0] = head;
//    System.arraycopy(tail, 0, this.dims, 1, tail.length);
//  }
//
//  public TensorShape(Shape shape) {
//    this.dims = dimsFromShape(shape);
//  }
//
//    /**
//     * Finds whether an indexed dimension in this TensorShape
//     * is known (i.e. != -1)
//     * @param i An index to check
//     * @return Whether dimension i is known.
//     */
//  public boolean isKnown(int i) {
//    return dims[i] != -1;
//  }
//
//    /**
//     * Throw an exception if dimension i is not known.
//     * @param i An index to check
//     * @throws IllegalStateException if dimension i is not known.
//     */
//  public void assertKnown(int i) {
//    if (!isKnown(i)) {
//      throw new IllegalStateException("Dimension " + i + " in shape needs to be known.");
//    }
//  }
//
//    /**
//     * Replace the last dimension of this shape,
//     * and return the modified shape.
//     * @param dim the new value for the last dimension
//     * @return A modified TensorShape with the new given last dimension.
//     */
//  public TensorShape replaceLast(long dim) {
//    return replace(this.dims.length - 1, dim);
//  }
//
//  public long size(int i) {
//    return this.dims[i];
//  }
//
//  public int numDimensions() {
//    return this.dims.length;
//  }
//
//  public int numElements() {
//    int prod = 1;
//    for (int i = 0; i < this.numDimensions(); i++) {
//      prod *= Math.abs(this.dims[i]);
//    }
//    return prod;
//  }
//
//  public TensorShape replace(int i, long dim) {
//    dims[i] = dim;
//    return this;
//  }
//
//  public TensorShape concatenate(long dim) {
//    this.dims = Keras.concatenate(this.dims, dim);
//    return this;
//  }
//
//  public TensorShape addToFront(long dim) {
//    this.dims = Keras.concatenate(dim, dims);
//    return this;
//  }
//
//  @Override
//  public boolean equals(Object obj) {
//    if (obj instanceof TensorShape) {
//      TensorShape ts  = (TensorShape) obj;
//      for (int i = 0; i < dims.length; i++) {
//        if (dims[i] != ts.dims[i]) return false;
//      }
//      return true;
//    }
//
//    return false;
//  }
//
//  public Shape toShape() {
//    return Shape.make(head(dims), tail(dims));
//  }
//}
