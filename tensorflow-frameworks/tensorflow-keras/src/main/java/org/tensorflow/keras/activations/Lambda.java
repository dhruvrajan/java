package org.tensorflow.keras.activations;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;

import java.util.function.BiFunction;

/**
 * Creates an `Activation` from an unnamed function.
 */
public class Lambda<T extends Number> extends Activation<T> {
    private BiFunction<Ops, Operand<T>, Operand<T>> activation;

    public Lambda(BiFunction<Ops, Operand<T>, Operand<T>> activation) {
        super();
        this.activation = activation;
    }

    @Override
    protected Operand<T> call(Ops tf, Operand<T> inputs) {
        return activation.apply(tf, inputs);
    }
}
