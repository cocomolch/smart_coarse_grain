


addpath('C:\Users\pgotthe\Documents\WSINematics\smart_coarse_grain\smart_coarse_grain')
addpath('C:\Users\pgotthe\Documents\Fibrosarcoma_in_vitro\FS_inVitro_Code')
%% select data

baseDir = 'C:\Users\pgotthe\Documents\WSINematics\tiles3D_Fibrosarkoma\tile_03_04\';
nameBase = 'out_tile_x003_y004';
tifName = 'Merged'

dirTo3DNematics = [baseDir,'out_tile_x003_y004_zapotocky_May3rd_3d' ,'.mat'];
dirTo3DNematics_directorMethod = [baseDir,'out_tile_x003_y004_directorMethod_06Thresh_May4th_3d.mat'];%out_tile_x003_y004_directorMethod_06Thresh_May4th_3d ; out_tile_x003_y004_directorMethod_May4th_3d

dirTo3DNematics_winding = [baseDir,'out_tile_x003_y004_winding_May3rd_3d' ,'.mat'];

dirTo3DNematics_2D = [baseDir, nameBase, '_2d_xy.mat'];

h5disp(dirTo3DNematics_2D)
h5disp(dirTo3DNematics)

%% get raw intensities

raw = tiffreadVolume([baseDir, tifName, '.tif']);   % (H, W, 3*D)
D   = size(raw, 3) / 3;
he_vol = single(reshape(raw, size(raw,1), size(raw,2), 3, D));
he_vol = permute(he_vol, [1 2 4 3]);   % → (H, W, D, 3)

bad_slices = registration_qualitycontrol(he_vol, 'ZDim', 3);


%% load data
%read data

charges3D_directorMethod = h5read(dirTo3DNematics_directorMethod, '/data/charges3D'); %out_tile_x003_y004_directorMethod_06Thresh_May4th_3d
charges3D_directorMethod = deleteRim(charges3D_directorMethod, 10);

charges3D = h5read(dirTo3DNematics, '/data/charges3D');
charges3D = deleteRim(charges3D, 20);
charges3D_winding = h5read(dirTo3DNematics_winding, '/data/charges3D');


S_3d = h5read(dirTo3DNematics, '/data/S_3d');

nx_2d = h5read(dirTo3DNematics_2D, '/data/nx_2d');
ny_2d = h5read(dirTo3DNematics_2D, '/data/ny_2d');


nx_2d = h5read(dirTo3DNematics_2D, '/data/nx_cg_2d');
ny_2d = h5read(dirTo3DNematics_2D, '/data/ny_cg_2d');

nx_cg_3d  = h5read(dirTo3DNematics, '/data/nx_cg');
ny_cg_3d = h5read(dirTo3DNematics, '/data/ny_cg');
nz_cg_3d = h5read(dirTo3DNematics, '/data/nz_cg');


winding_clean  = h5read(dirTo3DNematics_2D, '/data/winding_clean_2d');
winding_clean_mergedCloseDefects = clean_defect_map(winding_clean, 10); 


S_2d = h5read(dirTo3DNematics_2D, '/data/S_2d');


%% PLOT STUFF

%plot isosrufaces S<0.1
isoSurfaces = S_3d < 0.5;
margin = 20;  % adjust as needed
isoSurfaces =  deleteRim(isoSurfaces, margin);
volshow(isoSurfaces)

%plot defects


charges_clean =  deleteRim(charges, margin);
disclinations = bwareaopen(charges_clean, 3);
%disclinations = imdilate(abs(disclinations) > 0.2, strel('disk', 2));

disclinations = imgaussfilt(double((disclinations) > 0.2), 10) > 0.5;
disclinations(mean(he_vol, 3) > 230) = 0;

volshow(disclinations)


%% write s3D

% ── config ────────────────────────────────────────────────────────────────────
gif_path    = [baseDir, 'S3D_zoom.gif'];
mp4_path    = [baseDir, 'S3D_zoom.mp4'];   % ← PowerPoint-friendly
frame_delay = 0.2;        % seconds per frame  (= 1/frame_rate)
steps       = 20;
imageRange  = 400:1700;

% ── set up video writer ───────────────────────────────────────────────────────
frame_rate = round(1 / frame_delay);
vw = VideoWriter(mp4_path, 'MPEG-4');
vw.FrameRate = frame_rate;
vw.Quality   = 95;        % 0-100; 95 gives near-lossless at reasonable file size
open(vw);

% ── figure (fixed size so every frame is identical) ──────────────────────────
fig = figure(1); clf;
fig.Position = [100 100 900 900];   % fix pixel size → consistent video resolution

for slice = 1:150

    % ── pick slice (skip bad / blank) ─────────────────────────────────────────
    if ismember(slice, bad_slices) || mean(double(squeeze(he_vol(imageRange,imageRange,slice,:))), 'all') > 240
       currentDefects = S_3d(:,:,slice-1);
    else
        currentDefects = S_3d(:,:,slice);
    end

    % ── render ────────────────────────────────────────────────────────────────
     % find defects
   
  
    % subsample every 100th pixel
    [H, W] = size(currentDefects);
    [X, Y] = meshgrid(1:W, 1:H);
    mask   = mod(X, steps) == 0 & mod(Y, steps) == 0;
    figure(1); clf;
   
    imagesc(currentDefects); colorbar;
    clim([0 1])
     title(['z = ', num2str(slice*2), 'µm'])
    drawnow;


    % ── capture frame ─────────────────────────────────────────────────────────
    frame = getframe(fig);

    % ── write to MP4 ──────────────────────────────────────────────────────────
    writeVideo(vw, frame);

    % ── also write to GIF (optional — keep or remove) ─────────────────────────
    [gif_img, cmap] = rgb2ind(frame.cdata, 256);
    if slice == 1
        imwrite(gif_img, cmap, gif_path, 'gif', 'LoopCount', Inf, 'DelayTime', frame_delay);
    else
        imwrite(gif_img, cmap, gif_path, 'gif', 'WriteMode', 'append', 'DelayTime', frame_delay);
    end

end

% ── finalise ──────────────────────────────────────────────────────────────────
close(vw);
fprintf('Saved MP4 : %s\n', mp4_path);
fprintf('Saved GIF : %s\n', gif_path);

volshow(S_3d > 0.8)

%% plot disclinations with the director method

% first save rotating disclinations
createVolshowGIF(charges3D_directorMethod, [baseDir, 'rotating_defects_3d_disclinations_directorMethod_zoom.gif'], 'NumFrames', 100, 'FPS', 10);

volshow(charges3D_directorMethod)

% ── config ────────────────────────────────────────────────────────────────────
gif_path    = [baseDir, 'defects_3d_disclinations_directorMethod_zoom.gif'];
mp4_path    = [baseDir, 'defects_3d_disclinations_directorMethod_zoom.mp4'];   % ← PowerPoint-friendly
frame_delay = 0.4;        % seconds per frame  (= 1/frame_rate)
steps       = 20;
imageRange  = 400:1700;

% ── set up video writer ───────────────────────────────────────────────────────
frame_rate = round(1 / frame_delay);
vw = VideoWriter(mp4_path, 'MPEG-4');
vw.FrameRate = frame_rate;
vw.Quality   = 95;        % 0-100; 95 gives near-lossless at reasonable file size
open(vw);

% ── figure (fixed size so every frame is identical) ──────────────────────────
fig = figure(1); clf;
fig.Position = [100 100 900 900];   % fix pixel size → consistent video resolution
for slice = 1:149
    % ── pick slice ────────────────────────────────────────────────────────────
    if ismember(slice, bad_slices) || mean(double(squeeze(he_vol(imageRange,imageRange,slice,:))), 'all') > 240
        idx = slice - 1;
    else
        idx = slice;
    end
    
    he_2d   = squeeze(he_vol(imageRange, imageRange, idx, :));
    nx      = (squeeze(nx_cg_3d(imageRange,imageRange,idx)))';
    ny      = (squeeze(ny_cg_3d(imageRange,imageRange,idx)))';
    
    % Get the winding data for this slice
    defects = logical(charges3D_directorMethod(imageRange, imageRange, idx))';

    % ── render ────────────────────────────────────────────────────────────────
    figure(1); clf;
    set(gcf, 'Color', 'w'); % clean white background
    
    % 1. Plot Background (HE image)
    r = he_2d(:,:,1);
    g = he_2d(:,:,2);
    b = he_2d(:,:,3);

    r(defects) = 0;
    g(defects) = 0;
    b(defects) = 0;
    
    he_2d = cat(3, r, g, b);
    imshow(double(he_2d./max(he_2d(:)))); 
    hold on;
    
    
    % 3. Plot Vector Field (Quiver)
    [H, W] = size(nx);
    [X, Y] = meshgrid(1:W, 1:H);
    mask = mod(X, steps) == 0 & mod(Y, steps) == 0;
    
    q = quiver(X(mask), Y(mask), nx(mask), ny(mask), 0.5, 'k', 'ShowArrowHead', 'off');
    q.LineWidth = 0.5;
    

    title(['z = ', num2str(slice*2), 'µm (|grad n| defects)'])
    axis equal tight
    drawnow;

    % ── capture & write ───────────────────────────────────────────────────────
    frame = getframe(fig);
    writeVideo(vw, frame);
    
    % (GIF logic remains same using frame)
end

close(vw);

%% plot 3D winding DEFECTS WITH NATIVE AND DIRECTORS
% ── config ────────────────────────────────────────────────────────────────────
gif_path    = [baseDir, 'defects_3dwinding_zoom.gif'];
mp4_path    = [baseDir, 'defects_3dwinding_zoom.mp4'];   % ← PowerPoint-friendly
frame_delay = 0.4;        % seconds per frame  (= 1/frame_rate)
steps       = 20;
imageRange  = 400:1700;

% ── set up video writer ───────────────────────────────────────────────────────
frame_rate = round(1 / frame_delay);
vw = VideoWriter(mp4_path, 'MPEG-4');
vw.FrameRate = frame_rate;
vw.Quality   = 95;        % 0-100; 95 gives near-lossless at reasonable file size
open(vw);

% ── figure (fixed size so every frame is identical) ──────────────────────────
fig = figure(1); clf;
fig.Position = [100 100 900 900];   % fix pixel size → consistent video resolution
for slice = 1:149
    % ── pick slice ────────────────────────────────────────────────────────────
    if ismember(slice, bad_slices) || mean(double(squeeze(he_vol(imageRange,imageRange,slice,:))), 'all') > 240
        idx = slice - 1;
    else
        idx = slice;
    end
    
    he_2d   = squeeze(he_vol(imageRange, imageRange, idx, :));
    nx      = (squeeze(nx_2d(imageRange,imageRange,idx)))';
    ny      = (squeeze(ny_2d(imageRange,imageRange,idx)))';
    
    % Get the winding data for this slice
    defects = charges3D_winding(imageRange, imageRange, idx);

    % ── render ────────────────────────────────────────────────────────────────
    figure(1); clf;
    set(gcf, 'Color', 'w'); % clean white background
    
    % 1. Plot Background (HE image)
    imshow(double(he_2d./max(he_2d(:)))); 
    hold on;
    
    % 2. Overlay Defects Heatmap with Transparency
    hHeat = imagesc(defects);
    % AlphaData: 1 (visible) where abs > 0.2, else 0 (transparent)
    set(hHeat, 'AlphaData', abs(defects) > 0.2); 
    colormap(jet); % Or your preferred map
    clim([-1, 1]); % Fixed range for colorbar consistency
    colorbar;
    
    % 3. Plot Vector Field (Quiver)
    [H, W] = size(nx);
    [X, Y] = meshgrid(1:W, 1:H);
    mask = mod(X, steps) == 0 & mod(Y, steps) == 0;
    
    q = quiver(X(mask), Y(mask), nx(mask), ny(mask), 0.5, 'k', 'ShowArrowHead', 'off');
    q.LineWidth = 1.3;
    
    % 4. Plot Charge Markers (+ and -)
    % Find positive defects
    [p_rows, p_cols] = find(defects > 0.2);
    scatter(p_cols, p_rows, 40, 'red', 'filled', 'MarkerEdgeColor', 'k'); % Positive = Red
    
    % Find negative defects
    [n_rows, n_cols] = find(defects < -0.2);
    scatter(n_cols, n_rows, 40, 'blue', 'filled', 'MarkerEdgeColor', 'k'); % Negative = Blue

    title(['z = ', num2str(slice*4), ' (Winding Defects)'])
    axis equal tight
    drawnow;

    % ── capture & write ───────────────────────────────────────────────────────
    frame = getframe(fig);
    writeVideo(vw, frame);
    
    % (GIF logic remains same using frame)
end

close(vw);



%% plot 2D defects IGF/mov


% ── config ────────────────────────────────────────────────────────────────────
gif_path    = [baseDir, 'defects_2d_zoom.gif'];
mp4_path    = [baseDir, 'defects_2d_zoom.mp4'];   % ← PowerPoint-friendly
frame_delay = 0.4;        % seconds per frame  (= 1/frame_rate)
steps       = 20;
imageRange  = 400:1700;

% ── set up video writer ───────────────────────────────────────────────────────
frame_rate = round(1 / frame_delay);
vw = VideoWriter(mp4_path, 'MPEG-4');
vw.FrameRate = frame_rate;
vw.Quality   = 95;        % 0-100; 95 gives near-lossless at reasonable file size
open(vw);

% ── figure (fixed size so every frame is identical) ──────────────────────────
fig = figure(1); clf;
fig.Position = [100 100 900 900];   % fix pixel size → consistent video resolution

for slice = 1:149

    % ── pick slice (skip bad / blank) ─────────────────────────────────────────
    if ismember(slice, bad_slices) || mean(double(squeeze(he_vol(imageRange,imageRange,slice,:))), 'all') > 240
        he_2d = squeeze(he_vol(imageRange, imageRange, slice-1, :));
        [plus_rows,  plus_cols]  = find(winding_clean_mergedCloseDefects(imageRange,imageRange,slice-1) ==  0.5);
        [minus_rows, minus_cols] = find(winding_clean_mergedCloseDefects(imageRange,imageRange,slice-1) == -0.5);
        nx = (squeeze(nx_2d(imageRange,imageRange,slice - 1)))';
        ny = squeeze(ny_2d(imageRange,imageRange,slice - 1))';
    else
        he_2d = squeeze(he_vol(imageRange, imageRange, slice,   :));
        [plus_rows,  plus_cols]  = find(winding_clean_mergedCloseDefects(imageRange,imageRange,slice) ==  0.5);
        [minus_rows, minus_cols] = find(winding_clean_mergedCloseDefects(imageRange,imageRange,slice) == -0.5);
        nx = (squeeze(nx_2d(imageRange,imageRange,slice)))';
        ny = squeeze(ny_2d(imageRange,imageRange,slice))';
    end

    % ── render ────────────────────────────────────────────────────────────────
     % find defects
   
  
    % subsample every 100th pixel
    [H, W] = size(nx);
    [X, Y] = meshgrid(1:W, 1:H);
    mask   = mod(X, steps) == 0 & mod(Y, steps) == 0;
    figure(1); clf;
    title(['z = ', num2str(slice*4)])
    imshow(double(he_2d./max(he_2d(:))));
    hold on
    h = quiver(X(mask), Y(mask), nx(mask), ny(mask), ...
        0.5, ...
        'k', ...
        'ShowArrowHead', 'off');
    h.LineWidth = 1.3;
    axis equal tight
    hold on
    scatter(plus_rows,plus_cols, 50, "red", 'filled')
    scatter(minus_rows,minus_cols, 50, "blue", 'filled')
    drawnow;


    % ── capture frame ─────────────────────────────────────────────────────────
    frame = getframe(fig);

    % ── write to MP4 ──────────────────────────────────────────────────────────
    writeVideo(vw, frame);

    % ── also write to GIF (optional — keep or remove) ─────────────────────────
    [gif_img, cmap] = rgb2ind(frame.cdata, 256);
    if slice == 1
        imwrite(gif_img, cmap, gif_path, 'gif', 'LoopCount', Inf, 'DelayTime', frame_delay);
    else
        imwrite(gif_img, cmap, gif_path, 'gif', 'WriteMode', 'append', 'DelayTime', frame_delay);
    end

end

% ── finalise ──────────────────────────────────────────────────────────────────
close(vw);
fprintf('Saved MP4 : %s\n', mp4_path);
fprintf('Saved GIF : %s\n', gif_path);



%% plot 2D defects IGF/mov

% ── config ────────────────────────────────────────────────────────────────────
gif_path    = [baseDir, 'defects_3dZapotocky_zoom.gif'];
mp4_path    = [baseDir, 'defects_3dZapotocky_zoom.mp4'];   % ← PowerPoint-friendly
frame_delay = 0.4;        % seconds per frame  (= 1/frame_rate)
steps       = 20;
imageRange  = 400:1700;

% ── set up video writer ───────────────────────────────────────────────────────
frame_rate = round(1 / frame_delay);
vw = VideoWriter(mp4_path, 'MPEG-4');
vw.FrameRate = frame_rate;
vw.Quality   = 95;        % 0-100; 95 gives near-lossless at reasonable file size
open(vw);

% ── figure (fixed size so every frame is identical) ──────────────────────────
fig = figure(1); clf;
fig.Position = [100 100 900 900];   % fix pixel size → consistent video resolution

for slice = 1:149

    % ── pick slice (skip bad / blank) ─────────────────────────────────────────
    if ismember(slice, bad_slices) || mean(double(squeeze(he_vol(imageRange,imageRange,slice,:))), 'all') > 240
        he_2d = squeeze(he_vol(imageRange, imageRange, slice-1, :));
        [plus_rows,  plus_cols]  = find(charges3D(imageRange,imageRange,slice-1) > 0);
        nx = (squeeze(nx_2d(imageRange,imageRange,slice - 1)))';
        ny = squeeze(ny_2d(imageRange,imageRange,slice - 1))';
    else
        he_2d = squeeze(he_vol(imageRange, imageRange, slice,   :));
        [plus_rows,  plus_cols]  = find(charges3D(imageRange,imageRange,slice) > 0);
        nx = (squeeze(nx_2d(imageRange,imageRange,slice)))';
        ny = squeeze(ny_2d(imageRange,imageRange,slice))';
    end

    % ── render ────────────────────────────────────────────────────────────────
     % find defects
   
  
    % subsample every 100th pixel
    [H, W] = size(nx);
    [X, Y] = meshgrid(1:W, 1:H);
    mask   = mod(X, steps) == 0 & mod(Y, steps) == 0;
    figure(1); clf;
    title(['z = ', num2str(slice*4)])
    imshow(double(he_2d./max(he_2d(:))));
    hold on
    h = quiver(X(mask), Y(mask), nx(mask), ny(mask), ...
        0.5, ...
        'k', ...
        'ShowArrowHead', 'off');
    h.LineWidth = 1.3;
    axis equal tight
    hold on
    scatter(plus_rows,plus_cols, 50, "black", 'filled')
    drawnow;


    % ── capture frame ─────────────────────────────────────────────────────────
    frame = getframe(fig);

    % ── write to MP4 ──────────────────────────────────────────────────────────
    writeVideo(vw, frame);

    % ── also write to GIF (optional — keep or remove) ─────────────────────────
    [gif_img, cmap] = rgb2ind(frame.cdata, 256);
    if slice == 1
        imwrite(gif_img, cmap, gif_path, 'gif', 'LoopCount', Inf, 'DelayTime', frame_delay);
    else
        imwrite(gif_img, cmap, gif_path, 'gif', 'WriteMode', 'append', 'DelayTime', frame_delay);
    end

end

% ── finalise ──────────────────────────────────────────────────────────────────
close(vw);
fprintf('Saved MP4 : %s\n', mp4_path);
fprintf('Saved GIF : %s\n', gif_path);



%% plot native histo GIF

% ── config ────────────────────────────────────────────────────────────────────
gif_path    = [baseDir, 'nativeHistologyStack_zoom.gif'];
mp4_path    = [baseDir, 'nativeHistologyStack_zoom.mp4'];   % ← PowerPoint-friendly
frame_delay = 0.4;        % seconds per frame  (= 1/frame_rate)
steps       = 20;
imageRange  = 400:1700;

% ── set up video writer ───────────────────────────────────────────────────────
frame_rate = round(1 / frame_delay);
vw = VideoWriter(mp4_path, 'MPEG-4');
vw.FrameRate = frame_rate;
vw.Quality   = 95;        % 0-100; 95 gives near-lossless at reasonable file size
open(vw);

% ── figure (fixed size so every frame is identical) ──────────────────────────
fig = figure(1); clf;
fig.Position = [100 100 900 900];   % fix pixel size → consistent video resolution

for slice = 1:149

    % ── pick slice (skip bad / blank) ─────────────────────────────────────────
    if ismember(slice, bad_slices) || mean(double(squeeze(he_vol(imageRange,imageRange,slice,:))), 'all') > 240
        he_2d = squeeze(he_vol(imageRange, imageRange, slice-1, :));
    else
        he_2d = squeeze(he_vol(imageRange, imageRange, slice,   :));
    end

    % ── render ────────────────────────────────────────────────────────────────
    img = double(he_2d) / double(max(he_2d(:)));
    img = add_scalebar(img, 2, 1000, BarValue=0);

    imshow(img);
    title(['z = ', num2str(slice * 2), ' µm'], ...
          'FontSize', 16, 'FontWeight', 'bold');
    drawnow;

    % ── capture frame ─────────────────────────────────────────────────────────
    frame = getframe(fig);

    % ── write to MP4 ──────────────────────────────────────────────────────────
    writeVideo(vw, frame);

    % ── also write to GIF (optional — keep or remove) ─────────────────────────
    [gif_img, cmap] = rgb2ind(frame.cdata, 256);
    if slice == 1
        imwrite(gif_img, cmap, gif_path, 'gif', 'LoopCount', Inf, 'DelayTime', frame_delay);
    else
        imwrite(gif_img, cmap, gif_path, 'gif', 'WriteMode', 'append', 'DelayTime', frame_delay);
    end

end

% ── finalise ──────────────────────────────────────────────────────────────────
close(vw);
fprintf('Saved MP4 : %s\n', mp4_path);
fprintf('Saved GIF : %s\n', gif_path);