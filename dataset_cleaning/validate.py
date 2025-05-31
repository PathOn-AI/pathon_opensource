from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
import torch
from tqdm import tqdm
import argparse

def compare_episodes(dataset1, dataset2):
    try:
        # Create progress bar for the comparison
        with tqdm(total=len(dataset1), desc="Comparing frames", unit="frame") as pbar:
            for i in range(len(dataset1)):
                pbar.set_description(f"Checking frame {i}")
                
                frame1 = dataset1[i]
                frame2 = dataset2[i]
                
                # Compare actions
                assert torch.equal(frame1['action'], frame2['action']), \
                    f"Actions differ at frame {i}"
                
                # Compare observations
                assert torch.equal(frame1['observation.images.laptop'],
                                  frame2['observation.images.laptop']), \
                    f"Laptop images differ at frame {i}"
                
                assert torch.equal(frame1['observation.images.phone'],
                                  frame2['observation.images.phone']), \
                    f"Phone images differ at frame {i}"
                
                assert torch.equal(frame1['observation.state'],
                                  frame2['observation.state']), \
                    f"States differ at frame {i}"
                assert frame1['task'] == frame2['task'],\
                    f"Tasks differ at frame {i}"
                
                # Update progress bar
                pbar.update(1)
            
            print("✓ All frames match perfectly!")
        return True
            
    except AssertionError as e:
        print(f"✗ Comparison failed: {e}")
        return False

def main(new_repo_id: str, original_repo_id: str, new_episode: int, original_episode: int):
    print(f"New repo: {new_repo_id}")
    print(f"Original repo: {original_repo_id}")
    print(f"Comparing episode {new_episode} from new repo with episode {original_episode} from original repo\n")
    
    # Get metadata for the new dataset
    ds_meta = LeRobotDatasetMetadata(new_repo_id)
    
    # Display dataset information
    total_episodes: int = ds_meta.total_episodes
    print(f"Total number of episodes: {total_episodes}")
    avg_frames_per_episode: float = ds_meta.total_frames / total_episodes
    print(f"Average number of frames per episode: {avg_frames_per_episode:.3f}")
    fps: int = ds_meta.fps
    print(f"Frames per second used during data collection: {fps}")
    robot_type: str = ds_meta.robot_type
    print(f"Robot type: {robot_type}")
    camera_keys: list[str] = ds_meta.camera_keys
    print(f"keys to access images from cameras: {camera_keys=}\n")
    
    print(ds_meta.tasks)
    
    # Load the specified episode from new dataset
    print(f"\nLoading episode {new_episode} from new dataset...")
    new_dataset = LeRobotDataset(new_repo_id, episodes=[new_episode])
    print(f"New dataset task: {new_dataset[0]['task']}")
    print(f"New dataset keys: {new_dataset[0].keys()}")
    print(f"New dataset action: {new_dataset[0]['action']}")
    
    # Load the specified episode from original dataset
    print(f"\nLoading episode {original_episode} from original dataset...")
    original_dataset = LeRobotDataset(original_repo_id, episodes=[original_episode])
    print(f"Original dataset task: {original_dataset[0]['task']}")
    print(f"Original dataset keys: {original_dataset[0].keys()}")
    print(f"Original dataset action: {original_dataset[0]['action']}")
    
    # Compare datasets
    print(f"\nStarting comparison...")
    datasets_match = compare_episodes(
        new_dataset,
        original_dataset,
    )
    
    if datasets_match:
        print("\n🎉 Datasets are identical!")
    else:
        print("\n⚠️ Datasets have differences!")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare episodes between two LeRobot datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--new_repo_id', '-n', type=str, required=True,
                       help='Repository ID of the new/filtered dataset')
    parser.add_argument('--original_repo_id', '-o', type=str, required=True,
                       help='Repository ID of the original dataset')
    parser.add_argument('--new_episode', '-ne', type=int, required=True,
                       help='Episode number from the new dataset to compare')
    parser.add_argument('--original_episode', '-oe', type=int, required=True,
                       help='Episode number from the original dataset to compare')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.new_repo_id, args.original_repo_id, args.new_episode, args.original_episode)
