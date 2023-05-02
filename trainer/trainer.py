import platform
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from pprint import PrettyPrinter
from torch.utils.tensorboard import SummaryWriter
from tools.utils import setup_seed, AverageMeter, a2t, t2a
from tools.loss import BiDirectionalRankingLoss, TripletLoss, NTXent
from tools.info_loss import InFoNCELoss
from tools.make_csvfile import make_csv


from models.ASE_model import ASE
from data_handling.DataLoader import get_dataloader


def train(config):

    # setup seed for reproducibility
    setup_seed(config.training.seed)

    # set up logger
    exp_name = config.exp_name
    
    folder_name = '{}_freeze_{}_lr_{}_' \
                  'seed_{}'.format(exp_name, str(config.training.freeze),
                                             config.training.lr,
                                             config.training.seed)

    log_output_dir = Path('outputs', folder_name, 'logging')
    model_output_dir = Path('outputs', folder_name, 'models')
    log_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    if config.training.csv:
        csv_output_dir = Path('outputs', folder_name, 'csv')
        csv_output_dir.mkdir(parents=True, exist_ok=True)
   
    logger.remove()

    logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)
    logger.add(log_output_dir.joinpath('output.txt'), format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)

    main_logger = logger.bind(indent=1)

    # setup TensorBoard
    writer = SummaryWriter(log_dir=str(log_output_dir) + '/tensorboard')

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')

    # set up model
    if torch.cuda.is_available():
        device, device_name = ('cuda',torch.cuda.get_device_name(torch.cuda.current_device()))
    # elif torch.backends.mps.is_available():
    #     device, device_name = ('mps',platform.processor()) 
    else: 
        device, device_name = ('cpu', platform.processor())

    main_logger.info(f'Process on {device}:{device_name}')

    model = ASE(config)
    model = model.to(device)

    # set up optimizer and loss
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.training.lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=5,threshold=0.005,threshold_mode='abs',min_lr=0.000001,verbose=True)

    if config.training.loss == 'triplet':
        criterion = TripletLoss(margin=config.training.margin)
    
    elif config.training.loss == 'ntxent':
        criterion = NTXent()
    
    elif config.training.loss == 'weight':
        criterion = WeightTriplet(margin=config.training.margin)
        
    elif config.training.loss == 'infonce':
        criterion = InFoNCELoss()
        
    else: #config.training.loss == 'bidirect': 'contrastive'??
        criterion = BiDirectionalRankingLoss(margin=config.training.margin)
    

    # set up data loaders
    train_loader = get_dataloader('train', config)
    val_loader = get_dataloader('val', config)
    test_loader = get_dataloader('test', config)

    main_logger.info(f'Size of training set: {len(train_loader.dataset)}, size of batches: {len(train_loader)}')
    main_logger.info(f'Size of validation set: {len(val_loader.dataset)}, size of batches: {len(val_loader)}')
    main_logger.info(f'Size of test set: {len(test_loader.dataset)}, size of batches: {len(test_loader)}')
    main_logger.info(f'Total parameters: {sum([i.numel() for i in model.parameters()])}')

    ep = 1

    # resume from a checkpoint
    if config.training.resume:
        checkpoint = torch.load(config.path.resume_model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        ep = checkpoint['epoch']

    # training loop
    recall_sum = []

    for epoch in range(ep, config.training.epochs + 1):
        main_logger.info(f'Training for epoch [{epoch}]')

        epoch_loss = AverageMeter()
        start_time = time.time()
        model.train()

        for batch_id, batch_data in tqdm(enumerate(train_loader), total=len(train_loader)):

            audios, captions, audio_ids, _, _ = batch_data

            # move data to GPU
            audios = audios.to(device)
            audio_ids = audio_ids.to(device)

            audio_embeds, caption_embeds = model(audios, captions)

            loss = criterion(audio_embeds, caption_embeds, audio_ids)

            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad)
            optimizer.step()

            epoch_loss.update(loss.cpu().item())
        writer.add_scalar('train/loss', epoch_loss.avg, epoch)

        elapsed_time = time.time() - start_time

        # validation loop, validation after each epoch
        main_logger.info("Validating...")
        r1, r5, r10, mAP10, medr, meanr, val_loss = validate(val_loader, model, device, criterion=criterion)
        r_sum = r1 + r5 + r10
        recall_sum.append(r_sum)

        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/r@1', r1, epoch)
        writer.add_scalar('val/r@5', r5, epoch)
        writer.add_scalar('val/r@10', r10, epoch)
        #writer.add_scalar('val/r@50', r50, epoch)
        writer.add_scalar('val/mAP10', mAP10, epoch)
        writer.add_scalar('val/med@r', medr, epoch)
        writer.add_scalar('val/mean@r', meanr, epoch) 

        # save model
        if r_sum >= max(recall_sum):
            main_logger.info('Model saved.')
            torch.save({
                'model': model.state_dict(),
                'optimizer': model.state_dict(),
                'epoch': epoch,
                'config': config
            }, str(model_output_dir) + '/best_model.pth')

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {epoch_loss.avg:.3f},'
                         f'\ttime: {elapsed_time:.1f}, lr: {current_lr:.6f}.')

    # Training done, evaluate on evaluation set
    main_logger.info('Training done. Start evaluating.')
    best_checkpoint = torch.load(str(model_output_dir) + '/best_model.pth')
    model.load_state_dict(best_checkpoint['model'])
    best_epoch = best_checkpoint['epoch']
    main_logger.info(f'Best checkpoint occurred in {best_epoch} th epoch.')
    validate(test_loader, model, device, criterion=criterion, return_ranks=config.training.csv, csv_output_dir=csv_output_dir)
    main_logger.info('Evaluation done.')
    writer.close()

@torch.no_grad()
def validate(data_loader, model, device, criterion=None, return_ranks=False, csv_output_dir=None):

    val_logger = logger.bind(indent=1)
    model.eval()
    val_loss = AverageMeter()
    with torch.no_grad():
        # numpy array to keep all embeddings in the dataset
        audio_embs, cap_embs , audio_names_ = None, None, None

        for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
            audios, captions, audio_ids, indexs, audio_names = batch_data
            # move data to GPU
            audios = audios.to(device)

            audio_embeds, caption_embeds = model(audios, captions)

            if audio_embs is None:
                audio_embs = np.zeros((len(data_loader.dataset), audio_embeds.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), caption_embeds.size(1)))
                if return_ranks:
                    audio_names_ = np.array(['                                          ' for i in range(len(data_loader.dataset))])

            # Code for validation loss
            if criterion!=None:
                loss = criterion(audio_embeds, caption_embeds, audio_ids)
                val_loss.update(loss.cpu().item())

            audio_embs[indexs] = audio_embeds.cpu().numpy()
            cap_embs[indexs] = caption_embeds.cpu().numpy()
            if return_ranks:
                audio_names_[indexs] = np.array(audio_names)

        val_logger.info(f'Validation loss: {val_loss.avg :.3f}')
        # evaluate text to audio retrieval
        if return_ranks:
            r1, r5, r10, mAP10, medr, meanr, ranks, top5 = t2a(audio_embs, cap_embs, return_ranks=True)
            make_csv(audio_names_, top5, csv_output_dir=csv_output_dir)
        else:
            r1, r5, r10, mAP10, medr, meanr = t2a(audio_embs, cap_embs)

        val_logger.info('Caption to audio: r1: {:.2f}, r5: {:.2f}, '
                        'r10: {:.2f}, mAP10: {:.2f}, medr: {:.2f}, meanr: {:.2f}'.format(
                         r1, r5, r10, mAP10, medr, meanr))
        

        # evaluate audio to text retrieval
        r1_a, r5_a, r10_a, mAP10_a, medr_a, meanr_a = a2t(audio_embs, cap_embs)

        val_logger.info('Audio to caption: r1: {:.2f}, r5: {:.2f}, '
                        'r10: {:.2f}, mAP10: {:.2f}, medr: {:.2f}, meanr: {:.2f}'.format(
                         r1_a, r5_a, r10_a, mAP10_a, medr_a, meanr_a))

        return r1, r5, r10, mAP10, medr, meanr, val_loss.avg