import numpy as np
import cv2
import torch
torch.set_grad_enabled(False)

from utils.crop_as_in_dataset import ImageWriter
from utils import utils

from pathlib import Path

from tqdm import tqdm

from utils.crop_as_in_dataset import LatentPoseFaceCropper
from utils.Graphonomy.networks import deeplab_xception_transfer, graph
from utils.Graphonomy.exp.inference.inference_folder import flip_cihp
from torch.autograd import Variable

from PIL import Image

def string_to_valid_filename(x):
    return str(x).replace('/', '_')

if __name__ == '__main__':
    import logging, sys
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger('drive')

    import argparse
    arg_parser = argparse.ArgumentParser(
        description="Render 'puppeteering' videos, given a fine-tuned model and driving images.\n"
                    "Be careful: inputs have to be preprocessed by 'utils/preprocess_dataset.sh'.",
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('checkpoint_path', type=Path,
        help="Path to the *.pth checkpoint of a fine-tuned neural renderer model.")
    arg_parser.add_argument('data_root', type=Path,
        help="Driving images' source: \"root path\" that contains folders\n"
             "like 'images-cropped', 'segmentation-cropped-ffhq', or 'keypoints-cropped'.")
    arg_parser.add_argument('--images_paths', type=Path, nargs='+',
        help="Driving images' sources: paths to folders with driving images, relative to "
             "'`--data_root`/images-cropped' (note: here 'images-cropped' is the "
             "checkpoint's `args.img_dir`). Example: \"id01234/q2W3e4R5t6Y monalisa\".")
    arg_parser.add_argument('--destination', type=Path, required=True,
        help="Where to put the resulting videos: path to an existing folder.")
    args = arg_parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Will run on device '{device}'")

    # Initialize the model
    logger.info(f"Loading checkpoint from '{args.checkpoint_path}'")
    checkpoint_object = torch.load(args.checkpoint_path, map_location='cpu')

    import copy
    saved_args = copy.copy(checkpoint_object['args'])
    saved_args.finetune = True
    saved_args.inference = True
    saved_args.data_root = args.data_root
    saved_args.world_size = 1
    saved_args.num_workers = 1
    saved_args.batch_size = 1
    saved_args.device = device
    saved_args.bboxes_dir = Path("/non/existent/file")
    saved_args.prefetch_size = 4

    embedder, generator, _, running_averages, _, _, _ = \
        utils.load_model_from_checkpoint(checkpoint_object, saved_args)

    if 'embedder' in running_averages:
        embedder.load_state_dict(running_averages['embedder'])
    if 'generator' in running_averages:
        generator.load_state_dict(running_averages['generator'])

    embedder.train(not saved_args.set_eval_mode_in_test)
    generator.train(not saved_args.set_eval_mode_in_test)

    cropper = LatentPoseFaceCropper((saved_args.image_size, saved_args.image_size))

    # net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=20,
    #                                                                              hidden_layers=128,
    #                                                                              source_classes=7, )
    # net.load_source_model(torch.load('/home/lichnost/programming/work/ml/head/latent-pose/latent-pose/utils/Graphonomy/data/model/universal_trained.pth'))
    # net.cuda()
    #
    # # adj
    # adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    # adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)
    #
    # adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    # adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()
    #
    # cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    # adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    # adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()
    #
    # net.eval()

    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2) # set double buffer for capture
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print("Video capture at {} fps.".format(fps))

    def torch_to_opencv(image):
        image = image.permute(1,2,0).clamp_(0, 1).mul_(255).cpu().byte().numpy().copy()
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR, dst=image)


    camera = None
    if True:
        try:
            import pyfakewebcam

            stream_id = 1
            webcam_width = webcam_height = 256
            camera = pyfakewebcam.FakeWebcam(f'/dev/video{stream_id}', webcam_width, webcam_height)
            camera.print_capabilities()
            print(f'Fake webcam created on /dev/video{stream_id}.')
        except Exception as ex:
            print('Fake webcam initialization failed:')
            print(str(ex))

    while True:
        _, frame = video_capture.read()
        image_cropped, extra_data = cropper.crop_image(frame)

        # cv2.imwrite('image_cropped.jpg', image_cropped)

        input = torch.from_numpy(image_cropped.astype(np.float32) / 255.0)\
            .permute(2, 0, 1).unsqueeze(0).cuda()
        # outputs = net.forward(input, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
        # # outputs = (outputs[:1] + torch.flip(flip_cihp(outputs[1:]), dims=[-1, ])) / 2
        #
        # background_probability = 1.0 - outputs.softmax(1)[:, 0]  # `B x H x W`
        # background_probability = (background_probability * 255).round().byte().cpu().squeeze().numpy()
        # # background_probability = cv2.flip(background_probability, 1)
        # cv2.imwrite('segment.png', background_probability)

        data_dict = {
            'pose_input_rgbs' : input.unsqueeze(0)
        }

        embedder.get_pose_embedding(data_dict)
        generator(data_dict)

        result = torch_to_opencv(data_dict['fake_rgbs'][0])

        if camera is not None:
            camera.schedule_frame(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        # pose_driver = torch_to_opencv(data_dict['pose_input_rgbs'][0, 0])

        # frame_grid = np.concatenate((cv2.cvtColor(pose_driver, cv2.COLOR_RGB2BGR), result), axis=1)
        # cv2.imwrite('frame_grid.jpg', frame_grid)
        # cv2.imshow('win', frame_grid)
        # to_show = Image.fromarray(cv2.cvtColor(frame_grid, cv2.COLOR_BGR2RGB))
        # handle = to_show.show()
        # to_show.close()

    # for driver_images_path in args.images_paths:
    #     # Initialize the data loader
    #     saved_args.val_split_path = driver_images_path
    #     from dataloaders.dataloader import Dataloader
    #     logger.info(f"Loading dataloader '{saved_args.dataloader}'")
    #     dataloader = Dataloader(saved_args.dataloader).get_dataloader(saved_args, part='val', phase='val')
    #
    #     current_output_path = (args.destination / string_to_valid_filename(driver_images_path)).with_suffix('.avi')
    #     current_output_path.parent.mkdir(parents=True, exist_ok=True)
    #     image_writer = ImageWriter.get_image_writer(current_output_path)
    #
    #     for data_dict, _ in tqdm(dataloader):
    #         utils.dict_to_device(data_dict, device)
    #
    #         embedder.get_pose_embedding(data_dict)
    #         generator(data_dict)
    #
    #         def torch_to_opencv(image):
    #             image = image.permute(1,2,0).clamp_(0, 1).mul_(255).cpu().byte().numpy().copy()
    #             return cv2.cvtColor(image, cv2.COLOR_RGB2BGR, dst=image)
    #
    #         result = torch_to_opencv(data_dict['fake_rgbs'][0])
    #         pose_driver = torch_to_opencv(data_dict['pose_input_rgbs'][0, 0])
    #
    #         frame_grid = np.concatenate((pose_driver, result), axis=1)
    #         image_writer.add(frame_grid)
