// package org.tensorflow.keras.initializers;

// import org.tensorflow.Operand;
// import org.tensorflow.keras.utils.Keras;
// import org.tensorflow.op.Ops;
// import org.tensorflow.op.core.Assign;

// public abstract class Initializer {
//     /**
//      * Adds an `Assign` Op to the graph to initialize
//      * a tensorflow variable as specified by the initializer.
//      *
//      * @param tf Tensorflow Ops Accessor
//      * @param in Variable to initialize
//      * @return Assign Operand created
//      */
//     public <T extends Number> Assign<?> apply(Ops tf, Operand<?> in, Class<?> dtype) {
//         return tf.assign(in, this.initialize(tf, Keras.shapeOperand(tf, in.asOutput().shape()), dtype));
//     }

//     /**
//      * Returns a Tensor object initialized as
//      * specified by the initializer.
//      *
//      * @param tf    Tensorflow Ops Handle
//      * @param shape Shape of the tensor
//      */
//     public abstract Operand<?> initialize(Ops tf, Operand<Integer> shape, Class<?> dtype);
// }
