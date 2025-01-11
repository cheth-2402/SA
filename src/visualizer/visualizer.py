import os
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
import pdb
import warnings
from PIL import Image
import wandb
import torch.nn.functional as F
warnings.filterwarnings('ignore')

class Plotter_v2:
    
    def __init__(self, name, save_path, rows = 4, columns = 6):
        '''
        Args:
            name: the file will be saved at save_path/{name}_{ids}.png
            rows: rows in each image
            columns: rows in each image
        '''
        self.name = name
        self.save_path = save_path
        self.rows = rows
        self.columns = columns

        #current plot count and image id
        self.plot_id = 0
        self.image_id = -1

        #initate the plots
        self.init()

    def init(self):
        '''
        Saves current plot and also create new one
        '''
        if(self.plot_id > 0):
            self.save()
        
        #create new plot
        self.images = [[]]
        self.image_id += 1
        self.plot_id = 0

    def save(self):
        #need to assemble the images
        images = [np.concatenate(i, axis = 1) for i in self.images]
        images = np.concatenate(images, axis = 0)
        images = (255*images).astype('uint8')
        images = Image.fromarray(images)
        image_path = os.path.join(self.save_path, f'{self.name}_{self.image_id}.png')
        images.save(image_path, dpi=(1000, 1000))

    def add_plot(self, image, title = ''):
        '''
        Add plot to the last empty space
        '''
        #check if need to save
        
        if(self.plot_id == self.rows*self.columns):
            self.init()
        
        #find the corresponding row and column
        row_id = self.plot_id // self.columns
        if(len(image.shape) == 2):
            image = np.expand_dims(image, axis=-1)
        if(image.shape[-1] == 1):
            image = np.repeat(image, 3, axis = -1)
        image = np.pad(image, ((1,1),(1,1),(0,0)), 'constant', constant_values=(1,1))
        
        if(len(self.images) < row_id + 1):
            for i in range(row_id + 1 - len(self.images)):
                self.images.append([])
        self.images[row_id].append(image)
        
        #increase the plot id
        self.plot_id += 1


class Plotter:

    def __init__(self, name, save_path, rows = 4, columns = 6):
        '''
        Args:
            name: the file will be saved at save_path/{name}_{ids}.png
            rows: rows in each image
            columns: rows in each image
        '''
        self.name = name
        self.save_path = save_path
        self.rows = rows
        self.columns = columns

        #current plot count and image id
        self.plot_id = 0
        self.image_id = -1

        #initate the plots
        self.init()

    def init(self):
        '''
        Saves current plot and also create new one
        '''
        if(self.plot_id > 0):
            self.save()
        
        #create new plot
        self.fig, self.axes = plt.subplots(self.rows, self.columns)
        for axes_list in self.axes:
            for ax in axes_list:
                ax.set_xticks([])
                ax.set_yticks([])
        self.image_id += 1
        self.plot_id = 0

    def save(self):
        if(self.plot_id > 0):
            image_path = os.path.join(self.save_path, f'{self.name}_{self.image_id}.png')
            plt.savefig(image_path)
            plt.close()

    def add_plot(self, image, title = ''):
        '''
        Add plot to the last empty space
        '''
        #check if need to save
        
        if(self.plot_id == self.rows*self.columns):
            self.init()
        
        #find the corresponding row and column
        row_id = self.plot_id // self.columns
        col_id = self.plot_id % self.columns
        self.axes[row_id][col_id].imshow(image)
        self.axes[row_id][col_id].set_title(title)
        
        #increase the plot id
        self.plot_id += 1

    def new_row(self):

        #just change the plot id
        row_id = self.plot_id // self.columns
        self.plot_id = (row_id+1)*self.columns

class Visualizer:

    def __init__(self,args):
        self.args = args

    '''get image from tensor'''
    def image_from_tensor(self, image, gif = False):
        '''
        Args:
            image: 3 x H x W tensor or H x W
        Returns:
            output_image: detached numpy image H x W x 3
        '''
        output_image = image.detach().cpu()
        if(len(output_image.shape) == 3):
            output_image =  output_image.permute(1,2,0)
        output_image = output_image.numpy()
        if(gif):
            output_image = output_image*255
            output_image = output_image.astype(np.uint8)
        #this produces wrong conclusions
        else:
            output_image = output_image - np.min(output_image)
            output_image = output_image/np.max(output_image)    
        return output_image

    '''plot the results'''
    def plot_samples(self, results, plot_dir):

        
        '''plot the grids and add results to it'''
        if('generated' in results):
            #images history
            seq_generated = results['generated']
            seq_images = results['images']
            
            #get plotter for the image
            plotter = Plotter_v2('comp', plot_dir, rows = 4, columns = 6)

            #iterate over all sequences
            for frame_id in range(seq_generated.shape[0]):           
                    
                #plot the images
                orig_img = self.image_from_tensor(seq_images[frame_id])
                plotter.add_plot(orig_img,f'{frame_id}-orig')
                
                rec_img = self.image_from_tensor(seq_generated[frame_id])
                plotter.add_plot(rec_img,f'{frame_id}-gen')

                diff_img = np.abs(orig_img - rec_img)
                plotter.add_plot(diff_img, f'{frame_id}-diff')

            #save the final plot
            plotter.save()   

        if('generated_vae' in results):
            #images history
            seq_generated = results['generated_vae']
            seq_images = results['images']
            
            #get plotter for the image
            plotter = Plotter_v2('comp_vae', plot_dir, rows = 4, columns = 6)

            #iterate over all sequences
            for frame_id in range(seq_generated.shape[0]):           
                    
                #plot the images
                orig_img = self.image_from_tensor(seq_images[frame_id])
                plotter.add_plot(orig_img,f'{frame_id}-orig')
                
                rec_img = self.image_from_tensor(seq_generated[frame_id])
                plotter.add_plot(rec_img,f'{frame_id}-gen')

                diff_img = np.abs(orig_img - rec_img)
                plotter.add_plot(diff_img, f'{frame_id}-diff')

            #save the final plot
            plotter.save()        

        if(sum(['masks' == l for l in results.keys()])):
            
            
            seq_masks = results['masks']
            number_of_frames = seq_masks.shape[0]

            #plots in y direction
            nplots = seq_masks.shape[1] + 1
            plotter = Plotter_v2('masks', plot_dir, rows = 4, columns = nplots)
            
            for frame_id in range(number_of_frames):

                #plot the sequence form next row
                img = self.image_from_tensor(seq_images[frame_id])
                plotter.add_plot(img, f'{frame_id}')

                for object_id in range(nplots-1):
                    #pdb.set_trace()
                    img = self.image_from_tensor(seq_masks[frame_id,object_id])
                    plotter.add_plot(img)

            #save the final plot
            plotter.save() 

        if(sum(['content' in l for l in results.keys()])):
            
            seq_content = results['content']
            total_samples = seq_content.shape[0]

            #plots in y direction
            nplots = seq_content.shape[2] + 1
            plotter = Plotter_v2('content', plot_dir, rows = 4, columns = nplots)
            
            #iterate over all sequences and plotig masks
            for seq_id in range(total_samples):
                
                #plot the sequence form next row
                img = self.image_from_tensor(seq_images[seq_id])
                plotter.add_plot(img, f'{seq_id}')

                for object_id in range(nplots-1):
                    #pdb.set_trace()
                    img = self.image_from_tensor(seq_content[seq_id,object_id])
                    plotter.add_plot(img)

            #save the final plot
            plotter.save() 

    def tensor_to_video(self, tensor):
        #tensor: [B x T x 3 x H x W]
        tensor = (tensor + 0.5)/2*255
        tensor = tensor.clamp(0, 255)
        tensor = tensor.numpy().astype(np.uint8)
        return tensor

    def pad_video(self, video):
        #video: [B x T x 3 x H x W]
        pad = (1,1,1,1)
        video = F.pad(video, pad, "constant", 1)
        return video

    def make_grid_horizontal(self, video_list):
        #video: list of [[B] x T x 3 x H x W]
        video = torch.cat(video_list, dim = -1)
        return video
    
    def make_grid_verticle(self, video_list):
        #video: list of [[B] x T x 3 x H x W]
        video = torch.cat(video_list, dim = -2)
        return video

    def plot_wandb(self, results, mode, steps):

        #rollout videos
        #B x T x 3 x H x W
        if('generated' in results and 'images' in results):
            generated = self.pad_video(results['generated'][:6])
            true = self.pad_video(results['images'][:6])

            #stack horizontally
            videos = []
            for g,t in zip(generated, true):
                videos.append(self.make_grid_horizontal([t,g]))
            videos = self.make_grid_verticle(videos)
            videos = self.tensor_to_video(videos)
            videos = wandb.Video(videos, fps=4, caption=None)
            wandb.log({f"{mode}/samples": videos}, step = steps)

    '''
    move data to device
    '''
    def to_device(self, data):
        for i in data:
            if(isinstance(data[i], dict)):
                data[i] = self.to_device(data[i])
            else:
                data[i] = data[i].to(self.args.device)
        return data
    
    '''a comman function to visualize'''
    def visualize(self, mode, model, dataset, plot_dir, steps = 0):

        #just run the visualization for one round
        data = next(iter(dataset))
        data = self.to_device(data)
        model.eval()
        with torch.no_grad():
            #model.eval()
       
            #just run the model
            results = model.infer(data)
            
            #if wandb then visualize
            # if(self.args.wandb and (mode == 'long_rollout' or mode == 'test')):
            #     self.plot_wandb(results, mode, steps)

            #save the plots
            self.plot_samples(results, plot_dir)
