// package org.tensorflow.keras.activations;

// import org.junit.Test;
// import org.tensorflow.Tensor;
// import org.tensorflow.Tensors;

// import static org.tensorflow.TestUtils.assertTransformedTensorEquals;

// public class ActivationsTest {

//     @Test
//     public void testRelu() {
//         Tensor<Float> t1 = Tensors.create(new float[] {1.0f, 2.0f, 3.0f});
//         Tensor<Float> e1 = Tensors.create(new float[] {1.0f, 2.0f, 3.0f});

//         assertTransformedTensorEquals(Float.class, t1, e1, (tf, t) -> {
//             Activation<Float> relu = Activations.select(Activations.relu);
//             relu.build(tf, t.asOutput().shape(), Float.class);
//             return relu.apply(tf, t);
//         });

//         Tensor<Float> t2 = Tensors.create(new float[] {-1.0f, -2.0f, 3.0f});
//         Tensor<Float> e2 = Tensors.create(new float[] {0.0f, 0.0f, 3.0f});
//         assertTransformedTensorEquals(Float.class, t2, e2, (tf, t) -> {
//             Activation<Float> relu = Activations.select(Activations.relu);
//             relu.build(tf, t.asOutput().shape(), Float.class);
//             return relu.apply(tf, t);
//         });
//     }
// }
