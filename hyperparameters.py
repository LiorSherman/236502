preprocessor_params = {
    'dataset_path': "dataset/deutschl/test",
    'processed_path': "processed_dataset",
    'mapping_file': "processed_dataset/mapping.json",
    'one_string_dataset_file': "processed_dataset/full_processed_dataset",
    'seq_len': 64,
    'acceptable_durations': [
        0.25,  # 16th note
        0.5,  # 8th note
        0.75,
        1.0,  # quarter note
        1.5,
        2,  # half note
        3,
        4  # whole note
    ],
}

melody_generator_params = {
    'mapping_file': "processed_dataset/mapping.json",
    'seq_len': 64,
}
