//package org.tensorflow.keras.layers;
//
//import org.tensorflow.Operand;
//import org.tensorflow.Shape;
//import org.tensorflow.keras.initializers.Initializer;
//import org.tensorflow.op.Ops;
//import org.tensorflow.op.core.Assign;
//import org.tensorflow.op.core.Variable;
//
//import java.util.*;
//
///**
// * Base layer class.
// *
// * <p>A layer implements common neural network operations, such as convolution, batch norm, etc.
// * These operations require managing weights, losses, updates, and inter-layer connectivity.
// */
//public abstract class Layer<T extends Number> {
//    // Input() layer needs to access dtype and built.
//    protected Class<T> dtype;
//    protected boolean built;
//
//    private Map<String, LayerVariable<T>> weights;
//    private List<LayerVariable<T>> trainableWeights;
//    private List<LayerVariable<T>> nonTrainableWeights;
//
//    public Layer() {
//        this.built = false;
//    }
//
//    /**
//     * Override create(Ops) to add variables (weight tensors) to the layer.
//     * <p>
//     * The addWeight function and some tf ops require passing a Class<T> "dtype" object
//     * <p>
//     * To get the dtype of this layer in the build function, use Layer.getDtype()
//     *
//     * @param tf Tensorflow Ops accessor
//     */
//    protected abstract void build(Ops tf, List<Shape> inputShape);
//
//    /**
//     * Computes the output shape of the tensor returned by a Layer from the input tensor's shape
//     *
//     * @param inputShapes Shape of an input tensor to this layer
//     * @return Shape of the tensor that would be returned by `apply`.
//     */
//    public abstract List<Shape> computeOutputShape(List<Shape> inputShapes);
//
//    /**
//     * Defines the layer's logic, in terms of input operands, and variables.
//     *
//     * @param tf     Tensorflow Ops accessor.
//     * @param inputs A sequence of TF Operands
//     * @return The transformed input tensors, according to the layer's logic.
//     */
//    protected abstract List<Operand<?>> call(Ops tf, Operand<?>... inputs);
//
//    /**
//     * Internal wrapper for Layer.call
//     */
//    public final List<Operand<?>> apply(Ops tf, Operand<?>... inputs) {
//        if (!this.built) {
//            throw new IllegalStateException(
//                    "Layer.call() cannot be called before the layer is built (Layer.build())");
//        }
//
//        return this.call(tf, inputs);
//    }
//
//    protected final Variable<T> addWeight(Ops tf, String name, Shape shape, Initializer initializer, boolean trainable) {
//        Variable<T> variable = tf.variable(shape, dtype);
//        LayerVariable<T> layerVariable = new LayerVariable<>(variable, initializer.apply(tf, variable, variable.asOutput().dataType().asJavaClass()), trainable);
//
//        this.weights.put(name, layerVariable);
//        if (trainable) {
//            this.trainableWeights.add(layerVariable);
//        } else {
//            this.nonTrainableWeights.add(layerVariable);
//        }
//
//        return variable;
//    }
//
//    public boolean isBuilt() {
//        return this.built;
//    }
//}
//
//class LayerVariable<T> {
//    Variable<T> variable;
//    Assign<T> initializer;
//    boolean trainable;
//
//    public LayerVariable(Variable<T> variable, Assign<T> initializer, boolean trainable) {
//        this.variable = variable;
//        this.initializer = initializer;
//        this.trainable = trainable;
//    }
//}
