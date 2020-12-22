import sw_pathnetmod_tournament_eye_tracker

for i in range(5):
    parser = sw_pathnetmod_tournament_eye_tracker.get_parser()
    sw_pathnetmod_tournament_eye_tracker.main(parser.parse_args(
        ["0", "/mnt/data/DataSet/GazeCapture_pre/00010", "./", "--image_size", "224", "--batch_size", "64",
         "--epochs", "100", "--trained_model", "model/my_model.hdf5", "--transfer_all"])
    )

# for i in range(10):
#     parser = sw_pathnetmod_tournament_eye_tracker.get_parser()
#     sw_pathnetmod_tournament_eye_tracker.main(parser.parse_args(
#         ["0", "/mnt/data/DataSet/GazeCapture_pre/00010", "./", "--image_size", "224", "--batch_size", "64",
#          "--epochs", "100", "--trained_model", "model/my_model.hdf5", "--transfer_all", "--do_original"])
#     )
#
# for i in range(10):
#     parser = sw_pathnetmod_tournament_eye_tracker.get_parser()
#     sw_pathnetmod_tournament_eye_tracker.main(parser.parse_args(
#         ["0", "/mnt/data/DataSet/GazeCapture_pre/00010", "./", "--image_size", "224", "--batch_size", "64",
#          "--epochs", "100", "--trained_model", "model/my_model.hdf5", "--transfer_all"])
#     )
