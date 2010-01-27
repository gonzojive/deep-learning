clear all
path = 'bases/';
files = dir([path 'weights1*.dat']);
l = length(files);
for j=1:l
     file = [path, files(j).name];
     close all 
     W  = dlmread(file);
     figure, display_network(W');
     pause;  
end
close all

