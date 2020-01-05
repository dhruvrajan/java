package org.tensorflow;

import org.junit.Test;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat;

public class SimpleTest {
  @Test
  public void testSimple() {
    Ops tf = Ops.create();
    Variable<TFloat> variable = tf.variable(Shape.make(1, 2, 3), TFloat.DTYPE);
    tf.assign(variable, tf.zeros(tf.shape(tf.constant(new int[] {1, 2, 3})), TFloat.DTYPE));
  }
}
