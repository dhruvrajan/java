package org.tensorflow.keras.examples.mnist_refactor;

import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.data.Dataset;
import org.tensorflow.keras.datasets.MNIST;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Gradients;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.nn.Softmax;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.utils.Tuple2;
import org.tensorflow.utils.Pair;

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
      MNISTEagerClassifier mnist = new MNISTEagerClassifier();
      mnist.run();
    }

    public void run() {
        try {
            Ops tf = Ops.create();
            Pair<Tuple2<float[][][], float[][]>> data = MNIST.loadData();

            Dataset train = Dataset.fromTensorSlices(tf,
                    Arrays.asList(
                            Constant.create(tf.scope(), data.first().first()),
                            Constant.create(tf.scope(), data.first().second())),
                    Arrays.asList(TFloat.DTYPE, TFloat.DTYPE)
            ).batch(BATCH_SIZE);

            Variable<TFloat> weights = tf.variable(Shape.make(INPUT_SIZE, FEATURES), TFloat.DTYPE);
            Variable<TFloat> biases = tf.variable(Shape.make(FEATURES), TFloat.DTYPE);

            tf.assign(weights, tf.zeros(constArray(tf, INPUT_SIZE, FEATURES), TFloat.DTYPE));
            tf.assign(biases, tf.zeros(constArray(tf, FEATURES), TFloat.DTYPE));

            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                float epochAccuracy = 0;
                float epochLoss = 0;
                int batches = 0;
                for (List<Output<?>> batch : train) {
                    Operand<TFloat> images2D = tf.dtypes.cast(batch.get(0), TFloat.DTYPE);
                    Shape images2DShape = images2D.asOutput().shape();
                    long pixelsPerImage = images2DShape.size() / Math.abs(images2DShape.size(0));

                    Operand<TFloat> images = tf.reshape(images2D, tf.constant(new int[] { -1, (int) pixelsPerImage }));
                    Operand<TFloat> labels = tf.dtypes.cast(batch.get(1), TFloat.DTYPE);

                    Softmax<TFloat> softmax = tf.nn.softmax(tf.math.add(tf.linalg.matMul(images, weights), biases));
                    Mean<TFloat> crossEntropy =
                            tf.math.mean(
                                    tf.math.neg(tf.reduceSum(tf.math.mul(labels, tf.math.log(softmax)), constArray(tf, 1))),
                                    constArray(tf, 0));

                    Gradients gradients = tf.gradients(crossEntropy, Arrays.asList(weights, biases));
                    Constant<TFloat> alpha = Constant.create(tf.scope(), LEARNING_RATE);

                    tf.train.applyGradientDescent(weights, alpha, gradients.dy(0));
                    tf.train.applyGradientDescent(biases, alpha, gradients.dy(1));

                    Operand<TInt64> predicted = tf.math.argMax(softmax, Constant.create(tf.scope(), 1));
                    Operand<TInt64> expected = tf.math.argMax(labels, Constant.create(tf.scope(), 1));
                    Operand<TFloat> accuracy = tf.math.mean(
                            tf.dtypes.cast(tf.math.equal(predicted, expected), TFloat.DTYPE), tf.constant(0));

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

    private static Operand<TInt32> constArray(Ops tf, int... i) {
        return Constant.create(tf.scope(), i);
    }
}
