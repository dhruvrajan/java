package org.tensorflow.keras.examples.mnist;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.data.Dataset;
import org.tensorflow.data.OneShotIterator;
import org.tensorflow.keras.datasets.MNIST;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Gradients;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.nn.Softmax;
import org.tensorflow.op.train.ApplyGradientDescent;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.utils.Pair;

public class MNISTClassifier implements Runnable {
    private static final int INPUT_SIZE = 28 * 28;

    private static final float LEARNING_RATE = 0.2f;
    private static final int FEATURES = 10;
    private static final int BATCH_SIZE = 100;
    private static final int EPOCHS = 50;

    public static void main(String[] args) {
        new MNISTClassifier().run();
    }

    public void run() {
        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);

            // Load datasets
            Pair<Dataset> trainTestSplit;
            try {
                trainTestSplit = MNIST.loadDatasets(tf);
            } catch (IOException e) {
                System.out.println("Could not load dataset.");
                return;
            }
            Dataset trainDataset = trainTestSplit.first().batch(BATCH_SIZE);
            Dataset testDataset = trainTestSplit.second().batch(BATCH_SIZE);

            // Graph mode iteration with OneShotIterator
            OneShotIterator trainIterator = trainDataset.makeOneShotIterator();
            List<Output<?>> components = trainIterator.getComponents();
            Operand<TFloat32> images = components.get(0).expect(TFloat32.DTYPE);
            images = tf.reshape(images, tf.constant(new int[] {-1, 28 * 28}));
            Operand<TFloat32> labels = components.get(1).expect(TFloat32.DTYPE);

            // Declare, initialize weights
            Variable<TFloat32> weights = tf.variable(Shape.make(INPUT_SIZE, FEATURES), TFloat32.DTYPE);
            Assign<TFloat32> weightsInit = tf.assign(weights,
                    tf.zeros(constArray(tf, INPUT_SIZE, FEATURES), TFloat32.DTYPE));

            Variable<TFloat32> biases = tf.variable(Shape.make(FEATURES), TFloat32.DTYPE);
            Assign<TFloat32> biasesInit = tf.assign(biases, tf.zeros(constArray(tf, FEATURES), TFloat32.DTYPE));

            // 'Forward' call
            Softmax<TFloat32> softmax = tf.nn.softmax(tf.math.add(tf.linalg.matMul(images, weights), biases));
            Mean<TFloat32> crossEntropy = tf.math.mean(
                    tf.math.neg(tf.reduceSum(tf.math.mul(labels, tf.math.log(softmax)), constArray(tf, 1))),
                    constArray(tf, 0));

            // Gradient optimizer step
            Gradients gradients = tf.gradients(crossEntropy, Arrays.asList(weights, biases));
            Constant<TFloat32> alpha = tf.constant(LEARNING_RATE);
            ApplyGradientDescent<TFloat32> weightGradientDescent = tf.train.applyGradientDescent(weights, alpha,
                    gradients.dy(0));
            ApplyGradientDescent<TFloat32> biasGradientDescent = tf.train.applyGradientDescent(biases, alpha,
                    gradients.dy(1));

            // Compute accuracy metric
            Operand<TInt64> predicted = tf.math.argMax(softmax, tf.constant(1));
            Operand<TInt64> expected = tf.math.argMax(labels, tf.constant(1));
            Operand<TFloat32> accuracy = tf.math
                    .mean(tf.dtypes.cast(tf.math.equal(predicted, expected), TFloat32.DTYPE), tf.constant(0));

            try (Session session = new Session(graph)) {

                // Create iterator object
                session.runner()
                    .addTarget(weightsInit)
                    .addTarget(biasesInit)
                    .run();

                for (int i = 0; i < EPOCHS; i++) {
                    // reset iterator object
                    session.runner().addTarget(trainIterator.getMakeIteratorOp()).run();
                    
                    int batches = 0;
                    float epochAccuracy = 0;
                    while (true) {
                        try {
                            List<Tensor<?>> fetched = session.runner()
                                .addTarget(images)
                                .addTarget(labels)
                                .addTarget(weightGradientDescent)
                                .addTarget(biasGradientDescent)
                                .fetch(accuracy).run();

                            float batchAccuracy = fetched.get(0).floatValue();
                            epochAccuracy += batchAccuracy;
                            batches += 1;
                        } catch (IndexOutOfBoundsException e) {
                            break;
                        }
                    }

                    System.out.println("Epoch Accuracy " + i + ": " + epochAccuracy / batches);
                }
            }
        }
    }

    private static Operand<TInt32> constArray(Ops tf, int... i) {
        return tf.constant(i);
    }
}
