//package org.tensorflow.keras.initializers;
//
//import org.tensorflow.Operand;
//import org.tensorflow.keras.utils.Keras;
//import org.tensorflow.op.Ops;
//
//public class Zeros extends Initializer {
//  @Override
//  public <T extends Number> Operand<T> initialize(Ops tf, Operand<Integer> shape, Class<T> dtype) {
//    return org.tensorflow.op.core.Zeros.create(tf.scope(), shape, dtype);
//  }
//}
