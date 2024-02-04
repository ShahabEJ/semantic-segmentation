import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from src.data.dataset import BuildingDataset
from src.models.model import UNet
from src.utils.losses import get_loss_function
from src.utils.metrics import compute_metrics
from src.utils.config_loader import Config
from src.utils.logger import setup_logger
from src.utils.transformations import get_transformations

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, criterion, device, logger):
    model.eval()
    running_loss = 0.0
    accumulated_metrics = {}
    with torch.inference_mode():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            metrics = compute_metrics(outputs, masks)
        
            for name, value in metrics.items():
                if name in accumulated_metrics:
                    accumulated_metrics[name] += value
                else:
                    accumulated_metrics[name] = value

    avg_loss = running_loss / len(dataloader)
 
    for name in accumulated_metrics:
        accumulated_metrics[name] /= len(dataloader)

    metrics_log = ', '.join([f'{name}: {value:.4f}' for name, value in accumulated_metrics.items()])
    logger.info(f'Validation Loss: {avg_loss:.4f}, {metrics_log}')
    return avg_loss, accumulated_metrics


def main():
    config = Config.get_instance().config
    logger = setup_logger('train', 'logs/train.log')
    device = torch.device(config['device'])
    
    train_transform = get_transformations('train')
    val_transform = get_transformations('test')
    train_dataset = BuildingDataset(config['data']['train_path'], 'images', 'masks', transform=train_transform)
    val_dataset = BuildingDataset(config['data']['val_path'], 'images', 'masks', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    
    model = UNet(n_channels=config['model']['in_channels'], n_classes=config['model']['out_channels']).to(device)
    criterion = get_loss_function()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    for epoch in range(config['training']['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, logger)
        logger.info(f'Epoch {epoch+1}/{config["training"]["epochs"]}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Metrics: {val_metrics}')

        if epoch % config['logging']['save_checkpoint_interval'] == 0 or epoch == config['training']['epochs'] - 1:
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")
            logger.info(f'Checkpoint saved at epoch {epoch+1}')
