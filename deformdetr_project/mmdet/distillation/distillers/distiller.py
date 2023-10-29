import torch.nn as nn
import torch.nn.functional as F
import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict
from ..builder import DISTILLER,build_distill_loss
from collections import OrderedDict
from mmdet.models.builder import build_cl_head



@DISTILLER.register_module()
class Distiller(BaseDetector):
    """Base distiller for detectors.

    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 fgd_distill_cfg=None,
                 teacher_pretrained=None,
                 init_student=False,
                 is_layer_by_layer_distill=True,
                 use_teacher_group=True,
                 cl_head=dict(
                    type='MoCoCLHead',
                    img_channels=256,
                    pts_channels=256,
                    mid_channels=512,
                    loss_cl=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2))):

        super(Distiller, self).__init__()
        
        self.teacher = build_detector(teacher_cfg.model,
                                        train_cfg=teacher_cfg.get('train_cfg'),
                                        test_cfg=teacher_cfg.get('test_cfg'))
        self.init_weights_teacher(teacher_pretrained)
        self.teacher.eval()

        self.student= build_detector(student_cfg.model,
                                        train_cfg=student_cfg.get('train_cfg'),
                                        test_cfg=student_cfg.get('test_cfg'))
        if init_student:
            t_checkpoint = _load_checkpoint(teacher_pretrained)
            all_name = []
            for name, v in t_checkpoint["state_dict"].items():
                if name.startswith("backbone."):
                    continue
                else:
                    all_name.append((name, v))

            state_dict = OrderedDict(all_name)
            load_state_dict(self.student, state_dict)
        self.student.init_weights()
        
        self.distill_losses = nn.ModuleDict()
        self.fgd_distill_cfg = fgd_distill_cfg
        self.is_layer_by_layer_distill = is_layer_by_layer_distill
        self.use_teacher_group = use_teacher_group
        self.cl_head = build_cl_head(cl_head) if cl_head is not None else cl_head

        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())
        def regitster_hooks(student_module,teacher_module):
            def hook_teacher_forward(module, input, output):

                    self.register_buffer(teacher_module,output)
                
            def hook_student_forward(module, input, output):

                    self.register_buffer( student_module,output )
            return hook_teacher_forward,hook_student_forward
        
        for item_loc in fgd_distill_cfg:
            
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')

            self.register_buffer(student_module,None)
            self.register_buffer(teacher_module,None)

            hook_teacher_forward,hook_student_forward = regitster_hooks(student_module ,teacher_module )
            teacher_modules[item_loc.teacher_module].register_forward_hook(hook_teacher_forward)
            student_modules[item_loc.student_module].register_forward_hook(hook_student_forward)

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                self.distill_losses[loss_name] = build_distill_loss(item_loss)
        print("self.cl_head:", self.cl_head)
        if self.cl_head is not None:
            self.distill_losses['cl_head'] = self.cl_head

    def base_parameters(self):
        return nn.ModuleList([self.student,self.distill_losses])


    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self.student, 'roi_head') and self.student.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_bbox)
                or (hasattr(self.student, 'bbox_head') and self.student.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_mask)
                or (hasattr(self.student, 'mask_head') and self.student.mask_head is not None))

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')


    def forward_dummy(self, img):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """

        with torch.no_grad():
            self.teacher.eval()
            batch_input_shape = tuple(img[0].size()[-2:])
            for img_meta in img_metas:
                img_meta['batch_input_shape'] = batch_input_shape
 
            feat = self.teacher.extract_feat(img)
        #     teacher_bboxes_list, teacher_labels_list, teacher_all_stage_det_querys, query_embedding, positional_encoding \
        #                      = self.teacher.bbox_head.simple_test_teacher_distill(feat, img_metas, is_layer_by_layer_distill=self.is_layer_by_layer_distill)
           
        # student_loss, student_all_stage_det_querys, all_stage_pos_assigned_gt_inds, student_features = self.student.forward_train_distill( \
        #                 img, img_metas, teacher_bboxes=teacher_bboxes_list, teacher_labels=teacher_labels_list, \
        #                  is_layer_by_layer_distill=self.is_layer_by_layer_distill, **kwargs)

        """
        buffer_dict = dict(self.named_buffers())
        for item_loc in self.fgd_distill_cfg:
            
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')
            
            student_feat = buffer_dict[student_module]
            teacher_feat = buffer_dict[teacher_module]

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                
                student_loss[loss_name] = self.distill_losses[loss_name](student_feat,teacher_feat,kwargs['gt_bboxes'], img_metas)
        """
        
        return student_loss


    def forward_train(self, img, img_metas, **kwargs):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """

        with torch.no_grad():
            self.teacher.eval()
            batch_input_shape = tuple(img[0].size()[-2:])
            for img_meta in img_metas:
                img_meta['batch_input_shape'] = batch_input_shape
 
            tea_feat = self.teacher.extract_feat(img)
            teacher_bboxes_list, teacher_labels_list, teacher_all_stage_det_querys, query_embedding, positional_encoding, all_stage_cls_iou_score, all_stage_weight_query, all_stage_tea_assign_res \
                             = self.teacher.bbox_head.simple_test_teacher_distill(tea_feat, img_metas, is_layer_by_layer_distill=self.is_layer_by_layer_distill, gt_bboxes=kwargs['gt_bboxes'], gt_labels=kwargs['gt_labels'])
           
        student_loss, student_all_stage_det_querys, all_stage_pos_assigned_gt_inds, student_features = self.student.forward_train_distill( \
                        img, img_metas, teacher_bboxes=teacher_bboxes_list, teacher_labels=teacher_labels_list, \
                         is_layer_by_layer_distill=self.is_layer_by_layer_distill, **kwargs)

        stages = len(all_stage_cls_iou_score)
        loss_mse = nn.MSELoss(reduction='mean')
        for stage in range(stages):
            stage_cls_iou_score = all_stage_cls_iou_score[stage]
            stage_cls_iou_score = torch.stack(stage_cls_iou_score, dim=0)
            teacher_stage_det_querys = all_stage_weight_query[stage]
            teacher_stage_det_querys = torch.stack(teacher_stage_det_querys, dim=0)

            stage_loss = 0
            teacher_features = tea_feat
            # c_querys = torch.nn.functional.normalize(self.domain_covn(teacher_stage_det_querys), dim=-1)
            c_querys = torch.nn.functional.normalize(teacher_stage_det_querys, dim=-1)
            for fea_tea, fea_stu in zip(teacher_features, student_features):

                c_feats = torch.nn.functional.normalize(fea_tea, dim=1)
                mat = torch.einsum('bnc,bchw->bnhw', [c_querys, c_feats])

                mask = torch.einsum('bnhw,bn->bhw', [mat, stage_cls_iou_score]).clamp(min=1e-2)

                max_shu = torch.max(mask.flatten(1,2),dim=-1)[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
                mask = (mask / max_shu).unsqueeze(dim=1)

                stage_loss += loss_mse(mask* fea_tea, mask * fea_stu)
                student_loss[f'stage{stage}_relation_distill_loss'] = stage_loss

        if self.use_teacher_group:
            roi_losses = self.student.bbox_head.forward_train_teacher_group(
                student_features,
                img_metas,
                query_embeds=query_embedding,
                positional_encoding=positional_encoding,
                all_stage_tea_assign_res=all_stage_tea_assign_res,
                **kwargs)
            student_loss.update(roi_losses)

        if self.cl_head is not None:
            # TODO: if use the all 6 stages query 
            stages = len(teacher_all_stage_det_querys)
            for stage in range(stages):
                teacher_det_querys_stage = teacher_all_stage_det_querys[stage]
                student_det_querys_distill_stage = student_all_stage_det_querys[stage]
                pos_assigned_gt_inds_stage = all_stage_pos_assigned_gt_inds[stage]
                teacher_det_querys_distill_stage = [teacher_det_querys_stage[i][index] for i, index in enumerate(pos_assigned_gt_inds_stage)]
                # student_det_querys_distill_stage = torch.cat(student_det_querys_distill_stage)
                b, n, d = student_det_querys_distill_stage.size()
                student_det_querys_distill_stage = student_det_querys_distill_stage.reshape(b*n, d)
                teacher_det_querys_distill_stage = torch.cat(teacher_det_querys_distill_stage)
                
                relation_distill_loss_stage = self.cl_head(student_det_querys_distill_stage, teacher_det_querys_distill_stage)
                student_loss[f'stage{stage}_relation_distill_loss'] += relation_distill_loss_stage
        """
        buffer_dict = dict(self.named_buffers())
        for item_loc in self.fgd_distill_cfg:
            
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')
            
            student_feat = buffer_dict[student_module]
            teacher_feat = buffer_dict[teacher_module]

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                
                student_loss[loss_name] = self.distill_losses[loss_name](student_feat,teacher_feat,kwargs['gt_bboxes'], img_metas)
        """
        
        return student_loss
    
    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)
    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)
    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)


