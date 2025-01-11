import os
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
import pdb
import warnings
warnings.filterwarnings('ignore')

class Visualizer:

    def __init__(self,args):
        self.args = args
        print("dang it new one")

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
        else:
            output_image = output_image - np.min(output_image)
            output_image = output_image/np.max(output_image)    
        return output_image

    def get_plt(self):
        fig, axes = plt.subplots(height,width)
        for axes_list in axes:
            for ax in axes_list:
                ax.set_xticks([])
                ax.set_yticks([])
        return fig,axes

    def save_plt(self, plot_dir, image_count):
        plot_path = os.path.join(plot_dir, f'{image_count}.png')
        plt.savefig(plot_path)
        plt.close(fig)
    
    def plot_samples_help(self, sequences, titles, samples, height = 6, width = 6):

        #image counts
        image_count = 0
        
        #total samples 
        total_samples = len(sequences[0])
        ppp = height*width

        #iterate ovet total samples
        for sample_id in range(total_samples):
            
            #the current x and y coordinate
            curr_x, curr_y = 0,0
            fig, axes = self.get_plt(height, width)

            
            for seq_id, sequence in enumerate(sequences):
                if(curr_x*width + curr_y == ppp):
                    self.save_plt(plot_dir, image_count)

    '''plot the results'''
    def plot_samples(self, results, plot_dir):

        '''plot the grids and add results to it'''
        #images history
        seq_generated = results['history_generated'].unsqueeze(1)
        seq_images = results['history_images'].unsqueeze(1)
        seq_masks = results['history_masks'].unsqueeze(1)

        #images rollout
        rollout_generated = results['rollout_generated']
        rollout_images = results['rollout_images']
        rollout_masks = results['rollout_masks']

        #sequence images
        seq_generated = torch.cat((seq_generated,rollout_generated), dim = 1)
        seq_images = torch.cat((seq_images,rollout_images), dim = 1)
        seq_masks = torch.cat((seq_masks,rollout_masks), dim = 1)

        #initiate the plot counts
        
        image_count = 0
        total_samples = seq_images.shape[0]

        #iterate over all sequences
        for seq_id in range(total_samples):
            
            #plot the sequence form next row
            if(plot_count % 6 != 0):
                plot_count = plot_count + 6 - plot_count % 6

            for frame_id in range(seq_images.shape[1]): 

                #if overflow?
                if(plot_count == 0):

                    #create a plot
                    

                if(plot_count >= 24):

                    #save prev plot 
                    
                    
                    #create new plot
                    image_count += 1
                    fig, axes = plt.subplots(4,6)
                    for axes_list in axes:
                        for ax in axes_list:
                            ax.set_xticks([])
                            ax.set_yticks([])
                    plot_count = 0
                
                #plot the images
                orig_img = self.image_from_tensor(seq_images[seq_id][frame_id])
                axes[plot_count//6][plot_count%6].imshow(orig_img)
                axes[plot_count//6][plot_count%6].set_title(f'{seq_id}-{frame_id}-orig')
                plot_count += 1
                
                rec_img = self.image_from_tensor(seq_generated[seq_id][frame_id])
                axes[plot_count//6][plot_count%6].imshow(rec_img)
                axes[plot_count//6][plot_count%6].set_title(f'{seq_id}-{frame_id}-gen')
                plot_count += 1
                
                diff_img = np.abs(orig_img - rec_img)
                axes[plot_count//6][plot_count%6].imshow(diff_img)
                axes[plot_count//6][plot_count%6].set_title(f'{seq_id}-{frame_id}-diff')
                plot_count += 1
        
            #save the last plot
            plot_path = os.path.join(plot_dir, f'{image_count}.png')
            plt.savefig(plot_path)
            plt.close(fig)

            image_count = 0
            #no of objects
            no_of_objects = seq_masks[0].shape[1]

            #plots in y direction
            nplots = no_of_objects + 1

            if(sum(['masks' in l for l in results.keys()])):
                
                seq_masks = results['history_masks'].unsqueeze(1)
                rollout_masks = results['rollout_masks']
                seq_masks = torch.cat((seq_masks,rollout_masks), dim = 1)
            
                #iterate over all sequences and plotig masks
                for seq_id in range(total_samples):
                    
                    #plot the sequence form next row
                    if(seq_id % 4 == 0):
                        if(seq_id != 0):
                            #save prev plot 
                            plot_path = os.path.join(plot_dir, f'masks_{image_count}.png')
                            image_count += 1
                            plt.savefig(plot_path)
                            plt.close(fig)
                        fig, axes = plt.subplots(4,nplots)
                    img = self.image_from_tensor(seq_images[seq_id][0])
                    axes[seq_id%4,0].imshow(img, cmap='gray')
                    for object_id in range(nplots-1):
                        #pdb.set_trace()
                        img = self.image_from_tensor(seq_masks[seq_id,0,object_id])
                        axes[seq_id%4, object_id+1].imshow(img, cmap='gray')
                plot_path = os.path.join(plot_dir, f'masks_{image_count}.png')
                plt.savefig(plot_path)
                plt.close(fig)

            if(sum(['gmaps' in l for l in results.keys()])):
                
                seq_gmaps = results['history_gmaps'].unsqueeze(1)
                if(seq_id % 4 == 0):
                    if(seq_id != 0):
                        #save prev plot 
                        plot_path = os.path.join(plot_dir, f'gmap_{image_count}.png')
                        image_count += 1
                        plt.savefig(plot_path)
                        plt.close(fig)
                    fig, axes = plt.subplots(4,nplots)
                
                img = self.image_from_tensor(seq_images[seq_id,0])
                axes[seq_id%4,0].imshow(img, cmap='gray')
                
                for object_id in range(nplots-1):
                    #pdb.set_trace()
                    img = self.image_from_tensor(seq_gmaps[seq_id,0,object_id])
                    axes[seq_id%4, object_id+1].imshow(img, cmap='gray')
                
                plot_path = os.path.join(plot_dir, f'gmap_{image_count}.png')
                plt.savefig(plot_path)
                plt.close(fig)
                

    #the helper function for plot
    def plot_one_object(self, results, object_id, parameter, state, sequence_length, plot_dir):

        title = f'{parameter}_obj_{object_id}'
        data = []

        for seq_id in range(sequence_length):
            data.append(results[f'{title}_{state}_{seq_id}'])
        
        plt.plot(data)
        plt.title(f'{title} vs {state}')
        plt.ylabel(f"{parameter} error")
        plt.xlabel(f"{state}")
        plt.savefig(os.path.join(plot_dir, f'{title}_{state}.png'))
        plt.close()

    '''plot the results'''
    def plot(self, mode, results, plot_dir):

        #the rollout
        rollout = self.args[mode].rollout

        #need to plot on long rollout
        for state in ['history','rollout']:    
            if(state == 'history'):
                sequence_length = self.args.history
            else:
                sequence_length = self.args[mode].rollout
            for object_id in range(self.args.objects):
                for parameter in ['position', 'velocity']:
                    if(parameter == 'velocity' and state == 'history'):
                        continue
                    self.plot_one_object(results, object_id, parameter, state, sequence_length, plot_dir)


    def create_gif(self, model, dataset, rollout, gif_dir):

        #get the data
        data = next(iter(dataset))
        for k in data:
            data[k] = data[k][0:1,...]
        results = model.generate(data, rollout)
        images = results['rollout_generated']
        true_images = data['images']
        image_list = []
        true_image_list = []

        for i in range(rollout-1):
            image_i = self.image_from_tensor(images[0,i,...], gif = True)
            image_list.append(image_i)
            true_image_i = self.image_from_tensor(true_images[0,i,...], gif = True)
            true_image_list.append(true_image_i)
        generated_gif_path = os.path.join(gif_dir, "generate.gif")
        imageio.mimsave(generated_gif_path, image_list)
        true_gif_path = os.path.join(gif_dir, "true.gif")
        imageio.mimsave(true_gif_path, true_image_list)        

    '''a comman function to visualize'''
    def visualize(self, mode, model, dataset, plot_dir):

        #just run the visualization for one round
        data = next(iter(dataset))

        #just run the model
        results = model(data, mode = mode, vis_samples = True)
        for l in results:
            if('loss' in l):
                results[l] = results[l].sum(-1)
                
        
        #rollout len
        rollout = self.args[mode].rollout

        #save the plots
        self.plot_samples(results, plot_dir)
