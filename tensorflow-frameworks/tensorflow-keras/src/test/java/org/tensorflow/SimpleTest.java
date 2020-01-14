// package org.tensorflow;

// import org.junit.Test;
// import org.tensorflow.keras.utils.Keras;
// import org.tensorflow.op.Ops;
// import org.tensorflow.op.core.Assign;
// import org.tensorflow.op.core.Variable;
// import org.tensorflow.tools.Shape;
// import org.tensorflow.types.TFloat;

// import java.nio.FloatBuffer;
// import java.util.Arrays;
// import java.util.List;

// public class SimpleTest {
//   @Test
//   public void testSimple() {
//     try (Graph graph = new Graph()) {
//       Ops tf = Ops.create(graph);
//       Variable<TFloat> variable = tf.variable(Shape.make(1, 2, 3), TFloat.DTYPE);
//       Assign<TFloat> assign = tf.assign(variable, tf.zeros(tf.constant(new int[] {1, 2, 3}), TFloat.DTYPE));

//       try (Session session = new Session(graph)) {
//         session.runner().addTarget(assign).run();
//         List<Tensor<?>> outputs = session
//             .runner()
//             .fetch(variable)
//             .run();

//         printFloatTensor(outputs.get(0).expect(TFloat.DTYPE));

//       }
//     }
//   }

//   public static void printFloatTensor(Tensor<TFloat> tensor) {
//     FloatBuffer buffer = FloatBuffer.allocate((int) tensor.shape().size());
//     tensor.writeTo(buffer);
//     System.out.println(Arrays.toString(buffer.array()));
//   }

//   @Test
//   public void testSimpleEager() {
//     try (EagerSession eagerSession = EagerSession.create()) {
//       Ops tf = Ops.create(eagerSession);
//       Variable<TFloat> variable = tf.variable(Shape.make(1, 2, 3), TFloat.DTYPE);
//       Assign<TFloat> assign = tf.assign(variable, tf.zeros(tf.constant(new int[] {1, 2, 3}), TFloat.DTYPE));

//       Keras.printFloatTensor(variable.asOutput().tensor());
//     }


//   }
// }
