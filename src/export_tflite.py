import os
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def export_frozen_pb(model, checkpoints_dir):

    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    )

    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=checkpoints_dir,
        name="model.pb",
        as_text=False,
    )


if __name__ == "__main__":
    model_path = "0304_tf_2"
    model_path = os.path.join("checkpoints", model_path)

    model = tf.keras.models.load_model(model_path)
    model.summary()

    export_frozen_pb(model, model_path)

    with tf.compat.v1.Session() as sess:
        input_graph = os.path.join(model_path, "model.pb")

        input_arrays = ["x"]
        output_arrays = ["Identity", "Identity_1", "Identity_2", "Identity_3"]

        converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
            input_graph, input_arrays, output_arrays
        )

        tflite_model = converter.convert()
        open(os.path.join(model_path, "model.tflite"), "wb").write(tflite_model)
