function visualize_disclination(mat_3d_file_or_charges, tiff_file_or_vol, options)
% VISUALIZE_DISCLINATION  Render 3D disclination lines on H&E volume with volshow.
%
% Usage
% -----
%   % With file paths:
%   visualize_disclination('results_3d.mat', 'volume.tif')
%
%   % With matrices directly:
%   visualize_disclination(charges3D, he_vol, Skel3d = skel_3d)
%
%   % With name-value options:
%   visualize_disclination('results_3d.mat', 'volume.tif', ...
%       MumPerPxXY         = 2.0,  ...
%       MumPerPxZ          = 2.0,  ...
%       ExportGif          = true, ...
%       GifFile            = 'rotation.gif', ...
%       MinComponentVoxels = 100,  ...
%       TransparentRange   = [0.0 0.15; 0.94 1.0])
%
% Required arguments
% ------------------
%   mat_3d_file_or_charges   Path to HDF5 .mat file OR charges3D matrix directly.
%   tiff_file_or_vol         Path to raw multi-page H&E TIFF OR he_vol matrix directly.
%
% Name-value options                       Default      Description
% ─────────────────────────────────────────────────────────────────────────────
%   Skel3d              (logical 3D)        false(1,1,1) Required when arg1 is a matrix
%   MumPerPxXY          (double)            1.0          Lateral pixel size [um/px]
%   MumPerPxZ           (double)            1.0          Axial pixel size   [um/slice]
%   ChargeThresh        (double)            0            Show voxels with charges3D > this
%   MinComponentVoxels  (double)            50           Skip components smaller than this
%   MaxComponents       (double)            20           Max components to colour
%   HeOpacity           (double)            0.08         Max opacity of H&E volume [0-1]
%   RimMargin           (double)            10           Voxels to blank at each face
%   TransparentRange    (Nx2 double)        [0.0 0.15]   Intensity ranges made transparent
%   ExportGif           (logical)           false        Write a rotating GIF?
%   GifFile             (char)              ''           Output path (required if ExportGif=true)
%   GifNFrames          (double)            72           Frames for full 360 rotation
%   GifFps              (double)            20           Frames per second
%   GifElevation        (double)            25           Camera elevation [degrees]
%   FigureSize          (1x2 double)        [1200 900]   Figure width x height [px]
%   BackgroundColor                         'black'      Viewer background
%   LineColormap        (Nx3 double)        []           Empty = built-in red-yellow
%   Title               (char)              ''           Empty = auto-generated

    arguments
        mat_3d_file_or_charges                        % char path OR numeric matrix
        tiff_file_or_vol                              % char path OR numeric array
        options.Skel3d              (:,:,:) logical = false(1,1,1)
        options.MumPerPxXY          (1,1)   double  = 1.0
        options.MumPerPxZ           (1,1)   double  = 1.0
        options.ChargeThresh        (1,1)   double  = 0
        options.MinComponentVoxels  (1,1)   double  = 50
        options.MaxComponents       (1,1)   double  = 20
        options.HeOpacity           (1,1)   double  = 0.08
        options.RimMargin           (1,1)   double  = 10
        options.TransparentRange    (:,2)   double  = [0.0 0.15]
        options.ExportGif           (1,1)   logical = false
        options.GifFile             (1,:)   char    = ''
        options.GifNFrames          (1,1)   double  = 72
        options.GifFps              (1,1)   double  = 20
        options.GifElevation        (1,1)   double  = 25
        options.FigureSize          (1,2)   double  = [1200 900]
        options.BackgroundColor                     = 'black'
        options.LineColormap        (:,3)   double  = []
        options.Title               (1,:)   char    = ''
    end

    % ── Validate ─────────────────────────────────────────────────────────────
    if ischar(mat_3d_file_or_charges) || isstring(mat_3d_file_or_charges)
        assert(isfile(mat_3d_file_or_charges), ...
               'mat_3d_file not found:\n  %s', mat_3d_file_or_charges)
    end
    if ischar(tiff_file_or_vol) || isstring(tiff_file_or_vol)
        assert(isfile(tiff_file_or_vol), ...
               'tiff_file not found:\n  %s', tiff_file_or_vol)
    end
    assert(options.HeOpacity >= 0 && options.HeOpacity <= 1, ...
           'HeOpacity must be in [0, 1], got %.3f', options.HeOpacity)
    if options.ExportGif
        assert(~isempty(options.GifFile), ...
               'GifFile must be set when ExportGif = true')
    end
    if isnumeric(mat_3d_file_or_charges)
        assert(~all(options.Skel3d(:) == false, 'all') || isempty(options.Skel3d), ...
               'Skel3d must be provided when arg1 is a matrix')
    end

    % ── Load charges3D + skel_3d ──────────────────────────────────────────────
    if ischar(mat_3d_file_or_charges) || isstring(mat_3d_file_or_charges)
        fprintf('Loading %s ...\n', mat_3d_file_or_charges)
        charges3D = single(h5read(mat_3d_file_or_charges, '/data/charges3D'));
        skel_3d   = logical(h5read(mat_3d_file_or_charges, '/data/skel_3d'));
        % permute (Z,Y,X) from Python/HDF5 to (Y,X,Z) for MATLAB
        charges3D = permute(charges3D, [2 3 1]);
        skel_3d   = permute(skel_3d,   [2 3 1]);
    else
        charges3D = single(mat_3d_file_or_charges);
        skel_3d   = options.Skel3d;
    end

    % ── Load H&E volume ───────────────────────────────────────────────────────
    if ischar(tiff_file_or_vol) || isstring(tiff_file_or_vol)
        fprintf('Loading %s ...\n', tiff_file_or_vol)
        raw    = single(tiffreadVolume(tiff_file_or_vol));
        % TIFF stored as (H, W, 3*D) with RGB interleaved per z-slice
        D      = size(raw, 3) / 3;
        he_vol = reshape(raw, size(raw,1), size(raw,2), 3, D);
        he_vol = permute(he_vol, [1 2 4 3]);    % → (H, W, D, 3)
    else
        he_vol = single(tiff_file_or_vol);
    end

    % normalise per channel (works for both grayscale (H,W,D) and RGB (H,W,D,3))
    for c = 1:size(he_vol, 4)
        ch = he_vol(:,:,:,c);
        he_vol(:,:,:,c) = (ch - min(ch(:))) / (max(ch(:)) - min(ch(:)) + eps);
    end
    if ndims(he_vol) == 3   % grayscale — normalise whole volume
        he_vol = (he_vol - min(he_vol(:))) / (max(he_vol(:)) - min(he_vol(:)) + eps);
    end

    [H, W, D] = size(he_vol, 1, 2, 3);
    is_rgb    = size(he_vol, 4) == 3;
    fprintf('  Volume: H=%d  W=%d  D=%d  RGB=%d\n', H, W, D, is_rgb)

    % ── Rim masking ───────────────────────────────────────────────────────────
    m         = round(options.RimMargin);
    skel_3d   = blank_rim(skel_3d,   m);
    charges3D = blank_rim(charges3D, m);

    % ── Spatial reference (physical um axes) ──────────────────────────────────
    res = imref3d([H W D], ...
        options.MumPerPxXY, options.MumPerPxXY, options.MumPerPxZ);

    % ── Connected components of skeleton ──────────────────────────────────────
    fprintf('Labelling skeleton components...\n')
    CC            = bwconncomp(skel_3d, 26);
    comp_sizes    = cellfun(@numel, CC.PixelIdxList);
    [~, sort_idx] = sort(comp_sizes, 'descend');
    n_valid       = sum(comp_sizes(sort_idx) >= options.MinComponentVoxels);
    n_comp        = min(n_valid, round(options.MaxComponents));
    fprintf('  %d components >= %d voxels  (displaying %d)\n', ...
            n_valid, options.MinComponentVoxels, n_comp)

    % ── Build colour-coded overlay volume ─────────────────────────────────────
    overlay_vol = zeros(H, W, D, 'single');
    for k = 1 : n_comp
        idx              = CC.PixelIdxList{sort_idx(k)};
        overlay_vol(idx) = k / n_comp;
    end

    % ── Colourmaps ────────────────────────────────────────────────────────────
    n_colors = 256;
    cmap_he  = gray(n_colors);

    if isempty(options.LineColormap)
        % red (largest) → orange → yellow (smallest)
        cmap_disc = [ones(n_colors, 1), ...
                     linspace(0.1, 0.95, n_colors)', ...
                     zeros(n_colors, 1)];
    else
        cmap_disc = options.LineColormap;
        n_colors  = size(cmap_disc, 1);
    end

    % build alphamap then zero out transparent ranges
    alpha_he = linspace(0, options.HeOpacity, n_colors);
    for r = 1 : size(options.TransparentRange, 1)
        t_lo = round(options.TransparentRange(r,1) * (n_colors-1)) + 1;
        t_hi = round(options.TransparentRange(r,2) * (n_colors-1)) + 1;
        t_lo = max(1, min(t_lo, n_colors));
        t_hi = max(1, min(t_hi, n_colors));
        alpha_he(t_lo:t_hi) = 0;
    end

    alpha_disc = [0, ones(1, n_colors - 1)];   % bg transparent, lines opaque

    % ── Figure + viewer ───────────────────────────────────────────────────────
    fig = figure( ...
        'Name',     'Disclination Viewer', ...
        'Color',    'k', ...
        'Units',    'pixels', ...
        'Position', [100 100 options.FigureSize(1) options.FigureSize(2)]);

    viewer = viewer3d(fig, ...
        'BackgroundColor',    options.BackgroundColor, ...
        'BackgroundGradient', 'off', ...
        'Lighting',           'on');

    % ── H&E volume rendering ──────────────────────────────────────────────────
    if is_rgb
        hVol = volshow(he_vol, ...
            'Parent',         viewer, ...
            'Alphamap',       alpha_he, ...
            'RenderingStyle', 'VolumeRendering');
    else
        hVol = volshow(he_vol, ...
            'Parent',         viewer, ...
            'Colormap',       cmap_he, ...
            'Alphamap',       alpha_he, ...
            'RenderingStyle', 'VolumeRendering');
    end
    % ── Disclination overlay ──────────────────────────────────────────────────
    overlayVolume(hVol, overlay_vol, ...
        'Colormap', cmap_disc, ...
        'Alphamap', alpha_disc);

    % ── Axes cosmetics ────────────────────────────────────────────────────────
    ax = viewer.Axes;
    ax.XLabel.String = 'x [um]';  ax.XLabel.Color = 'w';
    ax.YLabel.String = 'y [um]';  ax.YLabel.Color = 'w';
    ax.ZLabel.String = 'z [um]';  ax.ZLabel.Color = 'w';
    ax.XColor = 'w';  ax.YColor = 'w';  ax.ZColor = 'w';

    if isempty(options.Title)
        ttl = sprintf('Disclination lines  |  %d components shown  |  charge threshold = %d', ...
                      n_comp, options.ChargeThresh);
    else
        ttl = options.Title;
    end
    title(ax, ttl, 'Color', 'w', 'FontSize', 13)

    view(ax, 45, options.GifElevation)
    camlight(ax, 'headlight')
    drawnow

    % ── GIF export ────────────────────────────────────────────────────────────
    if options.ExportGif
        export_rotating_gif(fig, ax, options)
    end

    fprintf('Done.\n')
end


%% ═══════════════════════════════════════════════════════════════════════════
%  Local functions
%  ═══════════════════════════════════════════════════════════════════════════

function vol = blank_rim(vol, m)
    vol(1:m,         :, :) = 0;
    vol(end-m+1:end, :, :) = 0;
    vol(:, 1:m,         :) = 0;
    vol(:, end-m+1:end, :) = 0;
    vol(:, :, 1:m        ) = 0;
    vol(:, :, end-m+1:end) = 0;
end


function export_rotating_gif(fig, ax, options)
    fprintf('Exporting GIF: %d frames @ %d fps -> %s\n', ...
            options.GifNFrames, options.GifFps, options.GifFile)

    angles  = linspace(0, 360, options.GifNFrames + 1);
    angles  = angles(1:end-1);
    delay_s = 1 / options.GifFps;

    for i = 1 : options.GifNFrames
        view(ax, angles(i), options.GifElevation)
        drawnow

        frame    = getframe(fig);
        im       = frame2im(frame);
        [A, map] = rgb2ind(im, 256);

        if i == 1
            imwrite(A, map, options.GifFile, 'gif', ...
                'LoopCount', Inf, 'DelayTime', delay_s)
        else
            imwrite(A, map, options.GifFile, 'gif', ...
                'WriteMode', 'append', 'DelayTime', delay_s)
        end

        if mod(i, 10) == 0
            fprintf('  frame %d / %d\n', i, options.GifNFrames)
        end
    end
    fprintf('  Saved: %s\n', options.GifFile)
end