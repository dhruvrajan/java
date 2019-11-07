package org.tensorflow.keras.activations;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;

import java.util.function.BiFunction;

/**
 * Helper functions to compute activations using a TF Ops object.
 */
public enum Activations {
    // All standard activations
    linear, sigmoid, tanh, relu, elu, selu, softmax, logSoftmax;

    /**
     * Create an `Activation` object given a type from the `Activations` enumeration.
     */
    public static <T extends Number> Activation<T> select(Activations type) {
        return new Lambda<>(getActivationFunction(type));
    }

    /**
     * Map from `Activations` enumeration to respective activation functions.
     */
    private static <T extends Number> BiFunction<Ops, Operand<T>, Operand<T>> getActivationFunction(Activations type) {
        switch (type) {
            case linear:
                return (tf, x) -> x;
            case sigmoid:
                return (tf, x) -> tf.math.sigmoid(x);
            case tanh:
                return (tf, x) -> tf.math.tanh(x);
            case relu:
                return (tf, x) -> tf.nn.relu(x);
            case elu:
                return (tf, x) -> tf.nn.elu(x);
            case selu:
                return (tf, x) -> tf.nn.selu(x);
            case softmax:
                return (tf, x) -> tf.nn.softmax(x);
            case logSoftmax:
                return (tf, x) -> tf.nn.logSoftmax(x);
            default:
                throw new IllegalArgumentException("Invalid Activation Type: " + type.name());
        }
    }
}
