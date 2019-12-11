// package org.tensorflow.keras.initializers;
//
// import org.tensorflow.Operand;
// import org.tensorflow.op.Ops;
// import org.tensorflow.op.core.Constant;
//
// public class Ones extends Initializer {
//  @Override
//  public <T extends Number> Operand<T> initialize(Ops tf, Operand<Integer> shape, Class<T> dtype) {
//    return tf.fill(shape, Constant.create(tf.scope(),1.0f, dtype));
//  }
// }
