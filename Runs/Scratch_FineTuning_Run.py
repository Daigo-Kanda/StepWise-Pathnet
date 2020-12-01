import finetuning
import scratch

for i in range(10):
    parser = scratch.getParser()
    scratch.main(parser.parse_args(["0", "/mnt/data/DataSet/GazeCapture_pre/00010", "./", "--image_size", "224",
                                    "--batch_size",
                                    "64", "--epochs", "100", "--trained_model", "model/my_model.hdf5"])
                 )

for i in range(10):
    parser = finetuning.getParser()
    finetuning.main(parser.parse_args(["0", "/mnt/data/DataSet/GazeCapture_pre/00010", "./", "--image_size", "224",
                                       "--batch_size",
                                       "64", "--epochs", "100", "--trained_model", "model/my_model.hdf5"])
                    )
