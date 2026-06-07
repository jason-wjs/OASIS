import os
import cv2
import json
import datetime
import numpy as np
import time
#from .rerun_visualizer import RerunLogger
from queue import Queue, Empty
from threading import Thread
import logging_mp
logger_mp = logging_mp.get_logger(__name__)

class EpisodeWriter():
    def __init__(self, task_dir, frequency=30, image_size=[640, 480], skip_json=False, text=None):
        """
        image_size: [width, height]
        text: optional task instruction string written into data.json text field.
        """
        logger_mp.info("==> EpisodeWriter initializing...\n")
        self.task_dir = task_dir
        self.frequency = frequency
        self.image_size = image_size
        self.skip_json = skip_json
        self._text_init = text

        self.data = {}
        self.episode_data = []
        self.item_id = -1
        self.episode_id = -1
        if os.path.exists(self.task_dir):
            episode_dirs = [episode_dir for episode_dir in os.listdir(self.task_dir) if 'episode_' in episode_dir]
            episode_last = sorted(episode_dirs)[-1] if len(episode_dirs) > 0 else None
            self.episode_id = 0 if episode_last is None else int(episode_last.split('_')[-1])
            logger_mp.info(f"==> task_dir directory already exist, now self.episode_id is:{self.episode_id}\n")
        else:
            os.makedirs(self.task_dir)
            logger_mp.info(f"==> episode directory does not exist, now create one.\n")
        self.data_info()
        self.text_desc()

        self.is_available = True  # Indicates whether the class is available for new operations
        # Initialize the queue and worker thread
        self.item_data_queue = Queue(-1)
        self.stop_worker = False
        self.need_save = False  # Flag to indicate when save_episode is triggered
        self.worker_thread = Thread(target=self.process_queue, daemon=True)
        self.worker_thread.start()
        logger_mp.info("==> EpisodeWriter initialized successfully.\n")


    def data_info(self, version='1.0.0'):
        self.info = {
                "version": "1.0.0" if version is None else version, 
                "date": datetime.date.today().strftime('%Y-%m-%d'),
                "author": "yzh" ,
                "image": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "depth": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "scene": ""
            }
    def text_desc(self):
        text = getattr(self, "_text_init", None)
        if isinstance(text, str) and text.strip():
            self.text = text
        else:
            self.text = {}

 
    def create_episode(self,scene, save_dir=None):
        if not self.is_available:
            #logger_mp.info("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
            return False  # Return False if the class is unavailable

        # Reset episode-related data and create necessary directories
        self.item_id = -1
        self.episode_data = []
        self.episode_id = self.episode_id + 1
        if save_dir:
            self.episode_dir = save_dir
        else:
            self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
        self.color_dir = os.path.join(self.episode_dir, 'colors')
        self.json_path = os.path.join(self.episode_dir, 'data.json')
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)

        self.is_available = False  # After the episode is created, the class is marked as unavailable until the episode is successfully saved
        logger_mp.info(f"==> New episode created: {self.episode_dir}")
        self.info['scene'] = scene
        return True  # Return True if the episode is successfully created
        
    def add_item(self, colors, states=None, actions=None, action_mimic=None, sim_state=None):
        # Increment the item ID
        self.item_id += 1
        # Create the item data dictionary
        item_data = {
            'idx': self.item_id,
            'colors': colors,
            'states': states,
            'actions': actions,
            'action_mimic': action_mimic,
            'sim_state': sim_state,
        }
        # Enqueue the item data
        self.item_data_queue.put(item_data)

    def process_queue(self):
        while not self.stop_worker or not self.item_data_queue.empty():
            # Process items in the queue
            try:
                item_data = self.item_data_queue.get(timeout=1)
                try:
                    self._process_item_data(item_data)
                except Exception as e:
                    logger_mp.info(f"Error processing item_data (idx={item_data['idx']}): {e}")
                self.item_data_queue.task_done()
            except Empty:
                pass
        
            # Check if save_episode was triggered
            if self.need_save and self.item_data_queue.empty():
                self._save_episode()

    def _process_item_data(self, item_data):
        """
        图像是一直在存并用路径代替，其余信息是等仿真结束了再存
        """
        idx = item_data['idx']
        colors = item_data.get('colors', {})

        # Save images
        if colors:
            for idx_color, (color_key, color) in enumerate(colors.items()):
                color_name = f'{str(idx).zfill(6)}_{color_key}.jpg'
                if not cv2.imwrite(os.path.join(self.color_dir, color_name), color):
                    logger_mp.info(f"Failed to save color image.")
                item_data['colors'][color_key] = os.path.join('colors', color_name)


        # Update episode data
        self.episode_data.append(item_data)


    def save_episode(self):
        """
        Trigger the save operation. This sets the save flag, and the process_queue thread will handle it.
        """
        self.need_save = True  # Set the save flag
        logger_mp.info(f"==> Episode saved start...")

    def _save_episode(self):
        """
        Save the episode data to a JSON file.
        """
        self.data['info'] = self.info
        self.data['text'] = self.text
        self.data['data'] = self.episode_data
        

        if not self.skip_json:
            with open(self.json_path, 'w', encoding='utf-8') as jsonf:
                jsonf.write(json.dumps(self.data, indent=4, ensure_ascii=False))
        else:
            print(f"==> [EpisodeWriter] Skip JSON saving for {self.episode_dir} (Handled externally)")

        self.need_save = False     # Reset the save flag
        self.is_available = True   # Mark the class as available after saving
        logger_mp.info(f"==> Episode saved successfully to {self.json_path}.")

    def clear_queue(self):
        """
        一键清除队列中所有尚未处理的数据，并重置当前 Episode 的缓存状态。
        """
        logger_mp.info("==> Cleaning up the data queue and resetting current episode data...")
        
        # 1. 清空队列中的所有项
        with self.item_data_queue.mutex:
            self.item_data_queue.queue.clear()
            # 通知所有正在 join() 的线程队列已空
            self.item_data_queue.all_tasks_done.notify_all()
            self.item_data_queue.unfinished_tasks = 0

        import shutil 
        if hasattr(self, 'episode_dir') and os.path.exists(self.episode_dir):
            try:
                shutil.rmtree(self.episode_dir)
                logger_mp.info(f"==> Deleted incomplete episode directory: {self.episode_dir}")
            except Exception as e:
                logger_mp.error(f"==> Failed to delete directory: {e}")
        # 2. 重置当前 Episode 的内存数据
        self.episode_data = []
        self.episode_id -= 1
        self.item_id = -1
        
        # 3. 恢复标志位
        self.need_save = False
        self.is_available = True
        
        logger_mp.info("==> Data queue and episode cache cleared successfully.")
        
    def close(self):
        """
        Stop the worker thread and ensure all tasks are completed.
        停止后台线程并确保所有任务完成
        """
        print("==>  Starting EpisodeWriter shutdown...")
        
        if not self.is_available:
            print("==>  Saving unfinished episode...")
            self.save_episode()
            while not self.is_available:
                time.sleep(0.01)
        
        # 停止工作线程
        # Stop worker thread
        print("==> Stopping worker thread...")
        self.stop_worker = True
        
        # 等待队列处理完成
        # Wait for queue processing to complete
        try:
            self.item_data_queue.join()
        except Exception as e:
            print(f"==>  Error waiting for queue completion: {e}")
        
        # 等待工作线程结束
        # Wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            print("==>  Waiting for worker thread to finish...")
            self.worker_thread.join(timeout=5.0)  # 5秒超时
            if self.worker_thread.is_alive():
                print("==>  Warning: Worker thread did not finish within timeout")
        
        print("==> EpisodeWriter shutdown completed")