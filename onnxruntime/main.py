import numpy as np
import onnxruntime as rt
import sys

DATA_PATH = '../data/npydata_zmj/'
ONNX_MODEL_PATH = '../models/sim_model.onnx'

def main(data_path=DATA_PATH, onnx_model_path=ONNX_MODEL_PATH):
        # read data
        X_test = np.load(data_path + 'X_test.npy')
        y_test = np.load(data_path + 'y_test.npy')

        # initialize onnx runtime inference session
        sess = rt.InferenceSession(onnx_model_path)

        # input & output names
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # input dimensions (important for debugging)
        input_dims = sess.get_inputs()[0].shape

        print(f"input_name: {input_name}        \
                output_name:{output_name}       \
                input_dims:{input_dims}")

        # infer
        from datetime import datetime
        results = []
        true_num = 0
        time_one_sample_start = np.array([])
        time_one_sample_end = np.array([])
        totaltime_200_samples_start = datetime.timestamp(datetime.now())
        for i, input in enumerate(X_test):
                input = X_test[i].astype(np.float32)
                input = np.reshape(input, (1, 1, 1500))
                time_one_sample_start = np.append(time_one_sample_start, datetime.timestamp(datetime.now()))
                result = sess.run([output_name], {input_name: input})[0]
                time_one_sample_end = np.append(time_one_sample_end, datetime.timestamp(datetime.now()))
                results.append(result)
                if np.argmax(result) == y_test[i]:
                        true_num  = true_num + 1

        totaltime_200_samples_end = datetime.timestamp(datetime.now())
        one_infer_time = np.mean(time_one_sample_end - time_one_sample_start)

        # statistic
        acc = true_num / len(y_test)
        print(f"acc = {acc}")
        print(f"200_samples_infer_time: {totaltime_200_samples_end - totaltime_200_samples_start}s")
        print(f"one_sample_infer_time: {one_infer_time}s")
        # print(results[0])



if __name__ == '__main__':
        # get arguments
        args = sys.argv[1:]
        if len(args) < 1:
                print("usage: ./main [DATASET_DIR_PATH] [ONNX_MODEL_PATH]")

        elif (args[1].find("onnx") == -1):
                print("usage: ./main [DATASET_DIR_PATH] [ONNX_MODEL_PATH]")   
        else:
                main(args[0], args[1])