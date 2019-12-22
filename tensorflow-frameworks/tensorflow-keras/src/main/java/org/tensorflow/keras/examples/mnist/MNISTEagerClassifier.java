package org.tensorflow.keras.examples.mnist;

import org.tensorflow.EagerSession;
import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Shape;
import org.tensorflow.data.Dataset;
import org.tensorflow.data.Pair;
import org.tensorflow.keras.datasets.MNIST;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Gradients;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.nn.Softmax;
import org.tensorflow.utils.TensorShape;
import org.tensorflow.data.Tuple2;

import java.io.IOException;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.List;

public class MNISTEagerClassifier implements Runnable {
    private static final int INPUT_SIZE = 28 * 28;

    private static final float LEARNING_RATE = 0.2f;
    private static final int FEATURES = 10;

    private static final int EPOCHS = 10;
    private static final int BATCH_SIZE = 100;

    public static void main(String[] args) {
        MNISTGraphClassifier mnist = new MNISTGraphClassifier();
        mnist.run();
    }

    public void run() {
        try (EagerSession session = EagerSession.getDefault()) {

            Ops tf = Ops.create(session);
            Tuple2<Pair<float[][][], float[][]>> data = MNIST.loadData();

            Dataset train = Dataset.fromTensorSlices(tf,
                    Arrays.asList(
                            Constant.create(tf.scope(), data.first().first()),
                            Constant.create(tf.scope(), data.first().second())),
                    Arrays.asList(Float.class, Float.class)
            ).batch(tf, BATCH_SIZE);

            Variable<Float> weights = tf.variable(Shape.make(INPUT_SIZE, FEATURES), Float.class);
            Variable<Float> biases = tf.variable(Shape.make(FEATURES), Float.class);

            tf.assign(weights, tf.zeros(constArray(tf, INPUT_SIZE, FEATURES), Float.class));
            tf.assign(biases, tf.zeros(Constant.create(tf.scope(), new int[]{FEATURES}), Float.class));

            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                float epochAccuracy = 0;
                float epochLoss = 0;
                int batches = 0;
                for (List<Output<?>> batch : train.asIterable(tf)) {
                    Operand<Float> images2D = tf.dtypes.cast(batch.get(0), Float.class);
                    TensorShape tensorShape = new TensorShape(images2D.asOutput().shape());
                    Operand<Float> images = tf.reshape(images2D,
                            Constant.create(tf.scope(), new int[]{-1, (int) (tensorShape.numElements() / Math.abs(tensorShape.size(0)))}));
                    Operand<Float> labels = tf.dtypes.cast(batch.get(1), Float.class);

                    Softmax<Float> softmax = tf.nn.softmax(tf.math.add(tf.linalg.matMul(images, weights), biases));
                    Mean<Float> crossEntropy =
                            tf.math.mean(
                                    tf.math.neg(tf.reduceSum(tf.math.mul(labels, tf.math.log(softmax)), constArray(tf, 1))),
                                    constArray(tf, 0));

                    Gradients gradients = tf.gradients(crossEntropy, Arrays.asList(weights, biases));
                    Constant<Float> alpha = Constant.create(tf.scope(), LEARNING_RATE);

                    tf.train.applyGradientDescent(weights, alpha, gradients.dy(0));
                    tf.train.applyGradientDescent(biases, alpha, gradients.dy(1));

                    Operand<Long> predicted = tf.math.argMax(softmax, Constant.create(tf.scope(), 1));
                    Operand<Long> expected = tf.math.argMax(labels, Constant.create(tf.scope(), 1));
                    Operand<Float> accuracy = tf.math.mean(tf.dtypes.cast(tf.math.equal(predicted, expected), Float.class), Constant.create(tf.scope(), 0));

                    epochAccuracy += accuracy.asOutput().tensor().floatValue();
                    epochLoss += crossEntropy.asOutput().tensor().floatValue();
                    batches++;
                }
                System.out.println("N BATCHES: " + batches);
                System.out.println("EPOCH #" + epoch + ":  acc = " + epochAccuracy / batches + " loss = " + epochLoss / batches);
            }
        } catch (IOException | NoSuchAlgorithmException e) {
            e.printStackTrace();
        }
    }

    private static Operand<Integer> constArray(Ops tf, int... i) {
        return Constant.create(tf.scope(), i);
    }
}
