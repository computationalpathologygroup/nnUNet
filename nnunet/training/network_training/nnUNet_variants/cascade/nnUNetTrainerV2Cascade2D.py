from nnunet.training.network_training.nnUNet_variants.loss_function.nnUNetTrainerV2_Loss_CEandDice_Weighted import \
    nnUNetTrainer_V2_Loss_CEandDice_Weighted


class nnUNetTrainerV2Cascade2D(nnUNetTrainer_V2_Loss_CEandDice_Weighted):

    def __init__(self, *args, **kwargs):
        super(nnUNetTrainerV2Cascade2D, self).__init__(*args, **kwargs)
        self.num_one_hot_encoded = kwargs.get("num_one_hot_encoded")

    def process_plans(self, plans):
        super().process_plans(plans)
        self.num_input_channels += self.num_classes - 2 if not self.num_one_hot_encoded else self.num_one_hot_encoded  # for seg from prev stage

    def setup_DA_params(self):
        super().setup_DA_params()

        self.data_aug_params["num_cached_per_thread"] = 2

        self.data_aug_params['move_last_seg_chanel_to_data'] = True
        self.data_aug_params['cascade_do_cascade_augmentations'] = True

        self.data_aug_params['cascade_random_binary_transform_p'] = 0.4
        self.data_aug_params['cascade_random_binary_transform_p_per_label'] = 1
        self.data_aug_params['cascade_random_binary_transform_size'] = (1, 8)

        self.data_aug_params['cascade_remove_conn_comp_p'] = 0.2
        self.data_aug_params['cascade_remove_conn_comp_max_size_percent_threshold'] = 0.15
        self.data_aug_params['cascade_remove_conn_comp_fill_with_other_class_p'] = 0.0

        # we have 2 channels now because the segmentation from the previous stage is stored in 'seg' as well until it
        # is moved to 'data' at the end
        self.data_aug_params['selected_data_channel_to_seg'] = [3]  # This will be 0 or 1 in selected_seg_channels
        self.data_aug_params['selected_seg_channels'] = [0, 1]

        self.data_aug_params['move_as_one_hot_to_data'] = True
        # needed for converting the segmentation from the previous stage to one hot
        self.data_aug_params['all_segmentation_labels'] = list(range(1, self.num_classes))

class nnUNetTrainerV2Cascade2DSoftMax(nnUNetTrainerV2Cascade2D):

    def setup_DA_params(self):
        super().setup_DA_params()

        # we have 2 channels now because the segmentation from the previous stage is stored in 'seg' as well until it
        # is moved to 'data' at the end
        self.data_aug_params['selected_data_channel_to_seg'] = list(range(3, self.num_classes + 1))  # This will be 0 or 1 in selected_seg_channels
        self.data_aug_params['selected_seg_channels'] = [0] + list(range(1, self.num_classes + 1))

        self.data_aug_params['move_as_one_hot_to_data'] = False
        # needed for converting the segmentation from the previous stage to one hot
        self.data_aug_params['all_segmentation_labels'] = list(range(1, self.num_classes + 1))
