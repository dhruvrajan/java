// package org.tensorflow;

// import org.junit.Test;
// import org.tensorflow.keras.utils.Keras;
// import org.tensorflow.op.Ops;
// import org.tensorflow.op.core.Constant;
// import org.tensorflow.op.core.Placeholder;
// import org.tensorflow.op.data.BatchDataset;

// import java.util.function.BiFunction;

// import static org.junit.Assert.assertFalse;
// import static org.junit.Assert.assertTrue;

// public class TestUtils {
//     public static <T> boolean transformedTensorEquals(Class<T> dtype, Tensor<T> tensor, Tensor<T> expect,
//                                                       BiFunction<Ops, Operand<T>, Operand<T>> transform) {

//         try (Graph graph = new Graph()) {
//             Ops tf = Ops.create(graph);

//             Operand<T> tensorOp = tf.placeholder(dtype, Placeholder.shape(Keras.shapeFromDims(tensor.shape())));
//             Operand<T> expectOp = tf.placeholder(dtype, Placeholder.shape(Keras.shapeFromDims(expect.shape())));
//             Operand<T> transformed = transform.apply(tf, tensorOp);
//             Operand<Boolean> equal = tf.math.equal(transformed, expectOp);

//             // Reduce 'equal' across each of its dimensions
//             long numDims = equal.asOutput().shape().numDimensions();
//             for (int i = 0; i < numDims; i++) {
//                 equal = tf.all(equal, Constant.create(tf.scope(), 0));
//             }

//             try (Session session = new Session(graph);
//                  Tensor<?> equalTensor = session.runner()
//                          .feed(tensorOp, tensor)
//                          .feed(expectOp, expect)
//                          .fetch(equal).run().get(0)) {

//                 return equalTensor.booleanValue();
//             }
//         }
//     }

//     public static <T> void assertTransformedTensorEquals(Class<T> dtype, Tensor<T> tensor, Tensor<T> expect,
//                                                           BiFunction<Ops, Operand<T>, Operand<T>> transform) {
//         assertTrue(transformedTensorEquals(dtype, tensor, expect, transform));
//     }

//     public static <T> void assertTransformedTensorNotEquals(Class<T> dtype, Tensor<T> tensor, Tensor<T> expect,
//                                                          BiFunction<Ops, Operand<T>, Operand<T>> transform) {
//         assertFalse(transformedTensorEquals(dtype, tensor, expect, transform));
//     }

//     @Test
//     public void transformedTensorCheck() {
//         Tensor<Float> tensor = Tensors.create(new float[][] {{1.0f, 3.0f}, {2.0f, 4.0f}});
//         Tensor<Float> expect = Tensors.create(new float[][] {{2.0f, 6.0f}, {4.0f, 8.0f}});

//         assertTransformedTensorNotEquals(Float.class, tensor, expect, (tf, t) -> t);
//         assertTransformedTensorEquals(Float.class, tensor, expect,
//                 (tf, t) -> tf.math.mul(t, Constant.create(tf.scope(), 2.0f)));
//     }
// }
