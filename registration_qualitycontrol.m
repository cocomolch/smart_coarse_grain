function bad_slices = registration_qc(vol, options)
% REGISTRATION_QC  Scroll through z-slices of a 3D volume, flag bad ones.
%
%   bad_slices = registration_qc(vol)
%   bad_slices = registration_qc(vol, um_per_px=2.0, bar_um=100)
%
% Input
% -----
%   vol          3-D numeric matrix  (Z x Y x X)  or  (Y x X x Z) — see ZDim
%
% Optional name-value
% -------------------
%   ZDim         which dimension is Z (default 1 → first dim is Z)
%   Colormap     colormap name (default 'gray')
%   ScaleBar     show scale bar on slices: true/false (default false)
%   um_per_px    pixel size [µm/px]  — needed if ScaleBar=true
%   bar_um       scalebar length [µm] — needed if ScaleBar=true
%
% Output
% ------
%   bad_slices   row vector of 1-based z-indices flagged as bad
%                (empty if none flagged, or if window is closed early)
%
% Controls
% --------
%   ← / →  arrow keys   previous / next slice
%   F  key               toggle flag on current slice
%   Slider               jump to any slice
%   Flag button          toggle flag on current slice
%   Done button          close GUI and return bad_slices

    arguments
        vol               {mustBeNumeric, mustBeNonempty}
        options.ZDim      (1,1) double {mustBePositive} = 1
        options.Colormap  (1,:) char   = 'gray'
        options.ScaleBar  (1,1) logical = false
        options.um_per_px (1,1) double = 1.0
        options.bar_um    (1,1) double = 100
    end

    % ── detect RGB (W,H,D,3) vs grayscale ────────────────────────────────────
    is_rgb = (ndims(vol) == 4 && size(vol, 4) == 3);

    % ── reorder so Z is first dim → (Z,Y,X) or (Z,Y,X,3) ───────────────────
    if is_rgb
        order = 1:3;                       % spatial dims only
        order(options.ZDim) = [];
        order = [options.ZDim, order, 4];  % keep channel last
    else
        order = 1:ndims(vol);
        order(options.ZDim) = [];
        order = [options.ZDim, order];
    end
    vol = permute(vol, order);

    nZ      = size(vol, 1);
    flagged = false(1, nZ);

    % ── normalise to [0 1] double for display ────────────────────────────────
    vol_d = double(vol);
    lo = min(vol_d(:));  hi = max(vol_d(:));
    if hi > lo
        vol_d = (vol_d - lo) / (hi - lo);
    end

    % ── build figure ──────────────────────────────────────────────────────────
    fig = figure( ...
        'Name',          'Registration QC', ...
        'NumberTitle',   'off', ...
        'Color',         [0.15 0.15 0.15], ...
        'Units',         'normalized', ...
        'Position',      [0.05 0.08 0.90 0.86], ...
        'KeyPressFcn',   @on_key, ...
        'CloseRequestFcn', @on_close);

    % ── axes (image) ──────────────────────────────────────────────────────────
    ax = axes('Parent', fig, ...
              'Units',  'normalized', ...
              'Position', [0.01 0.13 0.72 0.84], ...
              'Color',  [0 0 0]);
    colormap(ax, options.Colormap);
    h_img = imshow(get_slice(1), [], 'Parent', ax);
    axis(ax, 'image', 'off');

    % ── slice info label ──────────────────────────────────────────────────────
    h_title = uicontrol(fig, 'Style', 'text', ...
        'Units', 'normalized', 'Position', [0.01 0.965 0.72 0.030], ...
        'String', slice_str(1), ...
        'FontSize', 13, 'FontWeight', 'bold', ...
        'ForegroundColor', [1 1 1], 'BackgroundColor', [0.15 0.15 0.15], ...
        'HorizontalAlignment', 'center');

    % ── slider ────────────────────────────────────────────────────────────────
    sl_step = [1/(nZ-1+eps), 10/(nZ-1+eps)];
    h_slider = uicontrol(fig, 'Style', 'slider', ...
        'Units', 'normalized', 'Position', [0.01 0.04 0.72 0.04], ...
        'Min', 1, 'Max', nZ, 'Value', 1, ...
        'SliderStep', sl_step, ...
        'Callback', @on_slider, ...
        'BackgroundColor', [0.3 0.3 0.3]);

    % ── right panel controls ──────────────────────────────────────────────────
    panel_x = 0.75;

    % Flag button
    h_flag = uicontrol(fig, 'Style', 'pushbutton', ...
        'Units', 'normalized', 'Position', [panel_x 0.87 0.23 0.08], ...
        'String', 'Flag slice  [F]', ...
        'FontSize', 13, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.3 0.6 0.3], ...    % green = OK
        'ForegroundColor', [1 1 1], ...
        'Callback', @on_flag);

    % Previous / Next buttons
    uicontrol(fig, 'Style', 'pushbutton', ...
        'Units', 'normalized', 'Position', [panel_x 0.79 0.105 0.06], ...
        'String', '◀  Prev', ...
        'FontSize', 11, ...
        'BackgroundColor', [0.25 0.25 0.25], 'ForegroundColor', [1 1 1], ...
        'Callback', @(~,~) go(-1));

    uicontrol(fig, 'Style', 'pushbutton', ...
        'Units', 'normalized', 'Position', [panel_x+0.12 0.79 0.105 0.06], ...
        'String', 'Next  ▶', ...
        'FontSize', 11, ...
        'BackgroundColor', [0.25 0.25 0.25], 'ForegroundColor', [1 1 1], ...
        'Callback', @(~,~) go(+1));

    % Flagged slices list label
    uicontrol(fig, 'Style', 'text', ...
        'Units', 'normalized', 'Position', [panel_x 0.72 0.23 0.055], ...
        'String', 'Flagged slices:', ...
        'FontSize', 11, 'FontWeight', 'bold', ...
        'ForegroundColor', [1 0.4 0.4], 'BackgroundColor', [0.15 0.15 0.15], ...
        'HorizontalAlignment', 'left');

    h_list = uicontrol(fig, 'Style', 'listbox', ...
        'Units', 'normalized', 'Position', [panel_x 0.22 0.23 0.495], ...
        'String', {}, ...
        'FontSize', 10, ...
        'BackgroundColor', [0.1 0.1 0.1], 'ForegroundColor', [1 0.5 0.5], ...
        'Callback', @on_list_click);

    % Count label
    h_count = uicontrol(fig, 'Style', 'text', ...
        'Units', 'normalized', 'Position', [panel_x 0.175 0.23 0.04], ...
        'String', '0 slices flagged', ...
        'FontSize', 10, ...
        'ForegroundColor', [0.8 0.8 0.8], 'BackgroundColor', [0.15 0.15 0.15], ...
        'HorizontalAlignment', 'center');

    % Clear selected / Clear all
    uicontrol(fig, 'Style', 'pushbutton', ...
        'Units', 'normalized', 'Position', [panel_x 0.125 0.105 0.045], ...
        'String', 'Remove sel.', ...
        'FontSize', 9, ...
        'BackgroundColor', [0.3 0.25 0.25], 'ForegroundColor', [1 1 1], ...
        'Callback', @on_remove_sel);

    uicontrol(fig, 'Style', 'pushbutton', ...
        'Units', 'normalized', 'Position', [panel_x+0.12 0.125 0.105 0.045], ...
        'String', 'Clear all', ...
        'FontSize', 9, ...
        'BackgroundColor', [0.3 0.25 0.25], 'ForegroundColor', [1 1 1], ...
        'Callback', @on_clear_all);

    % Done button
    uicontrol(fig, 'Style', 'pushbutton', ...
        'Units', 'normalized', 'Position', [panel_x 0.04 0.23 0.07], ...
        'String', '✓  Done', ...
        'FontSize', 14, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.2 0.45 0.75], 'ForegroundColor', [1 1 1], ...
        'Callback', @on_done);

    % Keyboard hint
    uicontrol(fig, 'Style', 'text', ...
        'Units', 'normalized', 'Position', [panel_x 0.005 0.23 0.03], ...
        'String', '← → arrows to navigate  |  F to flag', ...
        'FontSize', 8, ...
        'ForegroundColor', [0.55 0.55 0.55], 'BackgroundColor', [0.15 0.15 0.15], ...
        'HorizontalAlignment', 'center');

    % ── state ─────────────────────────────────────────────────────────────────
    cur = 1;
    bad_slices = [];

    update_display();
    uiwait(fig);   % block until Done or window closed

    % ── callbacks ─────────────────────────────────────────────────────────────
    function on_slider(~, ~)
        cur = round(h_slider.Value);
        update_display();
    end

    function on_flag(~, ~)
        flagged(cur) = ~flagged(cur);
        update_display();
    end

    function on_key(~, evt)
        switch evt.Key
            case 'rightarrow',  go(+1);
            case 'leftarrow',   go(-1);
            case 'f',           flagged(cur) = ~flagged(cur); update_display();
        end
    end

    function go(delta)
        cur = max(1, min(nZ, cur + delta));
        update_display();
    end

    function on_list_click(~, ~)
        % double-click on list item → jump to that slice
        if strcmp(get(fig, 'SelectionType'), 'open')
            items = h_list.String;
            idx   = h_list.Value;
            if ~isempty(items)
                z = str2double(strtrim(strsplit(items{idx}, '–')));
                cur = z(1);
                update_display();
            end
        end
    end

    function on_remove_sel(~, ~)
        items = h_list.String;
        idx   = h_list.Value;
        if isempty(items), return; end
        z = str2double(strtrim(strsplit(items{idx}, '–')));
        flagged(z(1)) = false;
        update_display();
    end

    function on_clear_all(~, ~)
        flagged(:) = false;
        update_display();
    end

    function on_done(~, ~)
        bad_slices = find(flagged);
        uiresume(fig);
        delete(fig);
    end

    function on_close(~, ~)
        bad_slices = find(flagged);
        uiresume(fig);
        delete(fig);
    end

    % ── display update ────────────────────────────────────────────────────────
    function update_display()
        slc = get_slice(cur);

        % optionally burn scale bar (grayscale only)
        if options.ScaleBar && ~is_rgb
            slc = add_scalebar(slc, options.um_per_px, options.bar_um, ...
                               BarValue=1, Position='bottom-right');
        end

        set(h_img, 'CData', slc);
        set(h_slider, 'Value', cur);

        % title
        set(h_title, 'String', slice_str(cur));

        % flag button colour: red=flagged, green=OK
        if flagged(cur)
            set(h_flag, 'String',          '⚑  FLAGGED  [F to unflag]', ...
                        'BackgroundColor', [0.75 0.2 0.2]);
        else
            set(h_flag, 'String',          'Flag slice  [F]', ...
                        'BackgroundColor', [0.3 0.6 0.3]);
        end

        % highlight axes border
        if flagged(cur)
            set(ax, 'XColor', [0.9 0.2 0.2], 'YColor', [0.9 0.2 0.2], ...
                    'LineWidth', 4, 'Visible', 'on');
        else
            set(ax, 'Visible', 'off');
        end

        % rebuild flagged list
        bad_idx = find(flagged);
        list_str = arrayfun(@(z) sprintf('z = %d', z), bad_idx, ...
                            'UniformOutput', false);
        set(h_list, 'String', list_str, 'Value', min(h_list.Value, max(1,numel(list_str))));

        n = numel(bad_idx);
        if n == 0
            set(h_count, 'String', '0 slices flagged', 'ForegroundColor', [0.8 0.8 0.8]);
        else
            plural = 's'; if n == 1, plural = ''; end
            set(h_count, 'String', sprintf('%d slice%s flagged', n, plural), ...
                         'ForegroundColor', [1 0.5 0.5]);
        end

        drawnow;
    end

    function s = slice_str(z)
        s = sprintf('Slice  %d / %d', z, nZ);
        if flagged(z)
            s = [s, '    ⚑ FLAGGED'];
        end
    end

    % ── slice extractor (handles grayscale and RGB) ───────────────────────────
    function slc = get_slice(z)
        if is_rgb
            slc = squeeze(vol_d(z, :, :, :));   % (Y, X, 3)
        else
            slc = squeeze(vol_d(z, :, :));       % (Y, X)
        end
    end

end