function img = add_scalebar(img, um_per_px, bar_um, options)
% ADD_SCALEBAR  Burn a scale bar + µm label directly onto a 2D (or RGB) matrix.
%
% img = add_scalebar(img, um_per_px, bar_um)
% img = add_scalebar(img, 2.0, 100, BarValue=0, Position='bottom-left')
%
% Required
% --------
%   img         2D or MxNx3 numeric matrix
%   um_per_px   pixel size [µm/px]
%   bar_um      scalebar length [µm]
%
% Optional name-value
% -------------------
%   Position      'bottom-right'(default)|'bottom-left'|'top-right'|'top-left'
%   BarValue      pixel value for bar + label  (default: class max = white)
%   BarHeightFrac bar thickness as fraction of image height (default 0.012)
%   MarginFrac    margin from edge as fraction of image width (default 0.03)
%   FontScale     multiplier on auto font size (default 1.0)
%   LabelGap      gap between bar top and label [px] (default 3)

    arguments
        img           {mustBeNumeric}
        um_per_px     (1,1) double {mustBePositive}
        bar_um        (1,1) double {mustBePositive}
        options.Position      (1,:) char   = 'bottom-right'
        options.BarValue                   = []
        options.BarHeightFrac (1,1) double = 0.012
        options.MarginFrac    (1,1) double = 0.03
        options.FontScale     (1,1) double = 1.0
        options.LabelGap      (1,1) double = 3
    end

    [H, W, nCh] = size(img);

    % ── default bar value = class maximum (white) ─────────────────────────────
    if isempty(options.BarValue)
        switch class(img)
            case 'uint8',   options.BarValue = uint8(255);
            case 'uint16',  options.BarValue = uint16(65535);
            case 'single',  options.BarValue = single(1);
            otherwise,      options.BarValue = 1;
        end
    end
    barVal = double(options.BarValue);

    % ── geometry ──────────────────────────────────────────────────────────────
    bar_px   = round(bar_um / um_per_px);
    bar_h    = max(2, round(H * options.BarHeightFrac));
    margin_x = round(W * options.MarginFrac);
    margin_y = round(H * options.MarginFrac);

    pos = lower(options.Position);
    if contains(pos, 'right')
        x1 = W - margin_x - bar_px + 1;
    else
        x1 = margin_x + 1;
    end
    x2 = x1 + bar_px - 1;

    if contains(pos, 'bottom')
        y2 = H - margin_y;
    else
        y2 = margin_y + bar_h;
    end
    y1 = y2 - bar_h + 1;

    % ── draw bar ──────────────────────────────────────────────────────────────
    img(y1:y2, x1:x2, :) = cast(barVal, class(img));

    % ── render label into a tiny off-screen figure → extract pixel mask ───────
    label_str = sprintf('%g \xb5m', bar_um);   % µ = \xb5
    font_sz   = max(6, round(bar_h * 3.0 * options.FontScale));

    % Tight figure exactly the size of the bar width × enough height for text
    fig_w = bar_px;
    fig_h = round(font_sz * 2.5);

    fig = figure('Visible', 'off', ...
                 'Units',   'pixels', ...
                 'Position',[0 0 fig_w fig_h], ...
                 'Color',   [0 0 0]);           % black background
    ax = axes('Parent', fig, ...
              'Units',  'pixels', ...
              'Position',[0 0 fig_w fig_h], ...
              'XLim',   [0 1], 'YLim', [0 1]);
    axis(ax, 'off');

    text(ax, 0.5, 0.5, label_str, ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment',   'middle', ...
         'FontSize',   font_sz, ...
         'FontWeight', 'bold', ...
         'Color',      [1 1 1], ...     % white on black → threshold cleanly
         'Units',      'normalized');

    drawnow;
    frame   = getframe(ax);
    txt_rgb = double(frame.cdata);          % (fig_h, fig_w, 3)  uint8→double
    close(fig);

    % Resize if getframe returned slightly wrong dims
    if size(txt_rgb,1) ~= fig_h || size(txt_rgb,2) ~= fig_w
        txt_rgb = imresize(txt_rgb, [fig_h fig_w]);
    end

    % Binary mask: pixels where text was drawn (luminance > threshold)
    txt_mask = mean(txt_rgb, 3) > 30;      % (fig_h × fig_w) logical

    % ── stamp label above the bar, centred on it ──────────────────────────────
    gap  = round(options.LabelGap);
    ty1  = y1 - gap - fig_h;              % top of label patch
    ty2  = y1 - gap - 1;                  % bottom of label patch
    tx1  = x1;
    tx2  = x1 + fig_w - 1;

    % Clamp to image bounds
    % — vertical
    src_ry = 1:fig_h;
    if ty1 < 1
        clip  = 1 - ty1;
        ty1   = 1;
        src_ry = (clip+1):fig_h;
    end
    if ty2 > H
        ty2 = H;
        src_ry = src_ry(1 : ty2-ty1+1);
    end
    % — horizontal
    src_rx = 1:fig_w;
    if tx1 < 1
        clip  = 1 - tx1;
        tx1   = 1;
        src_rx = (clip+1):fig_w;
    end
    if tx2 > W
        tx2 = W;
        src_rx = src_rx(1 : tx2-tx1+1);
    end

    if ~isempty(src_ry) && ~isempty(src_rx)
        mask_crop = txt_mask(src_ry, src_rx);   % matching region of the glyph

        % Write barVal wherever the text glyph is set
        for c = 1:nCh
            ch = double(img(ty1:ty2, tx1:tx2, c));
            ch(mask_crop) = barVal;
            img(ty1:ty2, tx1:tx2, c) = cast(ch, class(img));
        end
    end
end