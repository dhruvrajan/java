/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/

// This class has been generated, DO NOT EDIT!

package org.tensorflow.op.nn;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;
import org.tensorflow.types.family.TNumber;

/**
 * Computes rectified linear 6 gradients for a Relu6 operation.
 * 
 * @param <T> data type for {@code backprops()} output
 */
public final class Relu6Grad<T extends TNumber> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Factory method to create a class wrapping a new Relu6Grad operation.
   * 
   * @param scope current scope
   * @param gradients The backpropagated gradients to the corresponding Relu6 operation.
   * @param features The features passed as input to the corresponding Relu6 operation, or
   * its output; using either one produces the same result.
   * @return a new instance of Relu6Grad
   */
  public static <T extends TNumber> Relu6Grad<T> create(Scope scope, Operand<T> gradients, Operand<T> features) {
    OperationBuilder opBuilder = scope.env().opBuilder("Relu6Grad", scope.makeOpName("Relu6Grad"));
    opBuilder.addInput(gradients.asOutput());
    opBuilder.addInput(features.asOutput());
    opBuilder = scope.applyControlDependencies(opBuilder);
    return new Relu6Grad<T>(opBuilder.build());
  }
  
  /**
   * The gradients:
   * `gradients * (features > 0) * (features < 6)`.
   */
  public Output<T> backprops() {
    return backprops;
  }
  
  @Override
  public Output<T> asOutput() {
    return backprops;
  }
  
  private Output<T> backprops;
  
  private Relu6Grad(Operation operation) {
    super(operation);
    int outputIdx = 0;
    backprops = operation.output(outputIdx++);
  }
}
