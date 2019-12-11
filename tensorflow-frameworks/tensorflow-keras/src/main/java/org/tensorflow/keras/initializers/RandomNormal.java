//package org.tensorflow.keras.initializers;
//
//
//import org.tensorflow.Operand;
//import org.tensorflow.op.core.Constant;
//import org.tensorflow.op.Ops;
//
//public class RandomNormal extends Initializer {
//    private float mean;
//    private float stdev;
//    private float p1;
//    private float p2;
//
//    public RandomNormal(float mean, float stdev, float p1, float p2) {
//        super();
//        this.mean = mean;
//        this.stdev = stdev;
//        this.p1 = p1;
//        this.p2 = p2;
//    }
//
//    @Override
//    public <T extends Number> Operand<T> initialize(Ops tf, Operand<Integer> shape, Class<T> dtype) {
//
//        return tf.random.parameterizedTruncatedNormal(
//                shape,
//                Constant.create(tf.scope(),this.mean, dtype),
//                Constant.create(tf.scope(),this.stdev, dtype),
//                Constant.create(tf.scope(),this.p1, dtype),
//                Constant.create(tf.scope(),this.p2, dtype)
//        );
//    }
//}
