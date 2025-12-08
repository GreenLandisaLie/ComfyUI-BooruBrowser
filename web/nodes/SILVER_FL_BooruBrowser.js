import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.SILVER_FL_BooruBrowser",
    async nodeCreated(node) {
        if (node.comfyClass === "SILVER_FL_BooruBrowser") {
            addBooruBrowserUI(node);
        }
    }
});

function addBooruBrowserUI(node) {
    // find widgets (they are auto-created by Comfy from INPUT_TYPES)
    const siteWidget = node.widgets.find(w => w.name === "site");
	const AND_tagsWidget = node.widgets.find(w => w.name === "AND_tags");
    const OR_tagsWidget = node.widgets.find(w => w.name === "OR_tags");
    const excludeWidget = node.widgets.find(w => w.name === "exclude_tags");
    const limitWidget = node.widgets.find(w => w.name === "limit");
    const pageWidget = node.widgets.find(w => w.name === "page");
    const safeWidget = node.widgets.find(w => w.name === "Safe");
    const questionableWidget = node.widgets.find(w => w.name === "Questionable");
    const explicitWidget = node.widgets.find(w => w.name === "Explicit");
	const orderWidget = node.widgets.find(w => w.name === "Order");
	
	const audioMutedWidget = node.widgets.find(w => w.name === "VIDEO_audio_muted");
	
	const thumbnailSizeWidget = node.widgets.find(w => w.name === "thumbnail_size");
	const thumbnailQualityWidget = node.widgets.find(w => w.name === "thumbnail_quality");
	const showFileExtWidget = node.widgets.find(w => w.name === "show_file_ext");
	
    const gelbooru_userWidget = node.widgets.find(w => w.name === "gelbooru_user_id");
    const gelbooru_apiWidget = node.widgets.find(w => w.name === "gelbooru_api_key");
	const danbooru_userWidget = node.widgets.find(w => w.name === "danbooru_user_id");
    const danbooru_apiWidget = node.widgets.find(w => w.name === "danbooru_api_key");
	const e621_userWidget = node.widgets.find(w => w.name === "e621_user_id");
    const e621_apiWidget = node.widgets.find(w => w.name === "e621_api_key");
	
    // hidden
    const selectedUrlWidget = node.widgets.find(w => w.name === "selected_url");
    const selectedIMGTagsWidget = node.widgets.find(w => w.name === "selected_img_tags");
	let currentTimeMSWidget = node.widgets.find(w => w.name === "current_time_ms");
    
    if (selectedUrlWidget) selectedUrlWidget.hidden = true;
    if (selectedIMGTagsWidget) selectedIMGTagsWidget.hidden = true;
	if (currentTimeMSWidget) currentTimeMSWidget.hidden = true;

    // UI sizing constants
	const MIN_WIDTH = 500;
    const MIN_HEIGHT = 1000;
    const TOP_PADDING = 750;
    const INPUT_SPACING = 6;
	const THUMB_SIZE = thumbnailSizeWidget ? thumbnailSizeWidget.value : 240;
    const THUMB_PADDING = 8;
    const SCROLLBAR_WIDTH = 32;

    // storage
    let posts = [];
    let thumbnails = {}; // map index -> ImageBitmap
    let scrollOffset = 0;
    let isDragging = false;
    let dragStartY = 0;
    let scrollStart = 0;
    let selectedIndex = -1;	
	
	// VIDEO ------------
	const FRAME_STEP = 5 / 30;
	
	let nodeSelected = false;
	let mouseOverThumbsGrid = false;
	
	let videoMode = false;           // are we displaying an active video in a thumbnail?
    let activeVideo = null;          // HTMLVideoElement used offscreen
    let activeVideoIndex = -1;       // which visible thumbnail index is activeVideo for
    let activeVideoFileURL = null;    // file id of active video
    let isVideoPlaying = false;
    let videoDuration = 0;
    let videoReady = false;
    let rafId = null;
    const videoFrameCanvas = document.createElement("canvas"); // offscreen temp canvas (for captures)
    let prevTimestampMsForWidget = 0;
	let isFullscreen = false;
	let videoContainer = null;  // DOM container for real video when maximizing
	let progressFs = null; // necessary global variable for the fullscreen progressBar so we can fix it not updating on manual seeking in non-fullscreen mode on transcoded videos
	let fullScreentimeDisplay = null; // same as above but for time display
	let activeIsGIF = false;
	let videoRotation = 0;
	
	// === AUDIO ===
	if (audioMutedWidget) {
		// --- 1️⃣ Watch for user changes in UI ---
		const original_callback = audioMutedWidget.callback;
		audioMutedWidget.callback = async function(value) {
			if (activeVideo) {
				activeVideo.muted = value;
				activeVideo.volume = value ? 0 : 1.0;
			}
			if (original_callback) original_callback(value);
		};
	}
	// -------------------
	
	
	
    // Create Search / Refresh buttons as widgets
    const searchButton = node.addWidget("button", "Search", null, async () => {
        await performSearch();
    });

    const prevButton = node.addWidget("button", "Prev Page", null, async () => {
        pageWidget.value = Math.max(1, (pageWidget.value || 1) - 1);
        await performSearch();
    });

    const nextButton = node.addWidget("button", "Next Page", null, async () => {
        pageWidget.value = (pageWidget.value || 1) + 1;
        await performSearch();
    });
	
	node.onSelected = function() {
        nodeSelected = true;
    };
    node.onDeselected = function() {
        nodeSelected = false;
    };
	function updateFullScreenTimeDisplay() {
		if (fullScreentimeDisplay) {
			const totalTime = formatTime(videoDuration);
			const currentTimeStr = videoReady && activeVideo ? formatTime(videoDuration > 0 ? activeVideo.currentTime : 0) : "0:00";
			fullScreentimeDisplay.textContent = `${currentTimeStr} / ${totalTime}`;
		}
	}
	
	function extractExtension(url) {
		const lastDot = url.lastIndexOf(".");
		if (lastDot === -1) return null;
	
		// take substring after last '.'
		let ext = url.substring(lastDot + 1);
	
		// strip query params if present
		const q = ext.indexOf("?");
		if (q !== -1) {
			ext = ext.substring(0, q);
		}
	
		// common booru extensions: jpg, jpeg, png, webp, gif, mp4, webm, etc
		if (ext.length >= 2 && ext.length <= 4) {
			return ext.toUpperCase();
		}
	
		return null;
	}

    // draw background (inputs are widgets; below we render the thumbnail area)
    node.onDrawBackground = function(ctx) {
        if (this.flags.collapsed) return;

        // Draw thumbnails area background
		const thumbAreaY = TOP_PADDING;
        const thumbAreaH = this.size[1] - TOP_PADDING - 10;
		
		
		// 💡 FIX 1: Clip the drawing to the thumbnail area
		ctx.save(); // Save the current context state
		ctx.beginPath();
		ctx.rect(0, thumbAreaY, this.size[0], thumbAreaH);
		ctx.clip(); // Set the clipping mask
		
		
        ctx.fillStyle = "#151515";
        ctx.fillRect(0, thumbAreaY, this.size[0], thumbAreaH);

        // Compute thumbnails grid
		const currentThumbSize = thumbnailSizeWidget ? thumbnailSizeWidget.value : THUMB_SIZE;
        const width = this.size[0] - SCROLLBAR_WIDTH - 10;
        const thumbsPerRow = Math.max(1, Math.floor(width / (currentThumbSize + THUMB_PADDING)));
        const rowHeight = currentThumbSize + THUMB_PADDING;

        // Draw each thumbnail
        posts.forEach((p, i) => {
            const row = Math.floor(i / thumbsPerRow);
            const col = i % thumbsPerRow;
            const x = 8 + col * (currentThumbSize + THUMB_PADDING);
            const y = thumbAreaY + THUMB_PADDING + row * rowHeight - scrollOffset;

            // skip if outside visible area
            if (y + currentThumbSize < thumbAreaY || y > thumbAreaY + thumbAreaH) return;

            // background
            ctx.fillStyle = "#252526";
            roundRect(ctx, x, y, currentThumbSize, currentThumbSize, 6);
            ctx.fill();
			
			if (videoMode && i === activeVideoIndex && activeVideo && videoReady) {
                try {
                    // Draw the current video frame into the thumbnail area
                    ctx.drawImage(activeVideo, x + 2, y + 2, currentThumbSize - 4, currentThumbSize - 4);
                } catch (e) {
                    // fallback to thumbnail image if drawImage fails
                    const bm = thumbnails[i];
                    if (bm) {
                        try {
                            ctx.drawImage(bm, x + 2, y + 2, currentThumbSize - 4, currentThumbSize - 4);
                        } catch {}
                    } else {
                        ctx.fillStyle = "#2d2d30";
                        ctx.fillRect(x + 2, y + 2, currentThumbSize - 4, currentThumbSize - 4);
                    }
                }
                // draw controls overlay
                drawVideoControls(ctx, x, y, currentThumbSize);
            } else {
                // normal thumbnail rendering (image or placeholder)
                const bm = thumbnails[i];
                if (bm) { // non-active/selected thumbnails that are already available
                    try {
                        ctx.drawImage(bm, x + 2, y + 2, currentThumbSize - 4, currentThumbSize - 4);
						
						const ext = extractExtension(p.file_url);
						if (ext && showFileExtWidget.value) {
							let extColor = "white";
							extColor = ext === "GIF" ? "#e98f00" // GIF
								// VIDEOS
								: (ext === "GIFV" || ext === "MP4" || ext === "WEBM" || ext === "M4V" || ext ==="MP4" || ext ==="M4V" || ext ==="WEBM" 
									|| ext ==="MOV" || ext ==="QT" || ext ==="AVI" || ext ==="MKV" || ext ==="FLV" || ext ==="F4V" 
									|| ext ==="WMV" || ext ==="MPG" || ext ==="MPEG" || ext ==="M2TS" || ext ==="3GP" || ext ==="3G2") ? "#003e73"
								// LOSSY/HIGH COMPRESSION IMG EXT - webp supports lossless compression but we are more likely to find lossy ones on boorus
								: (ext === "JPG" || ext === "JPEG" || ext === "WEBP" || ext === "AVIF") ? "yellow"
								// LOSSLESS/High-Efficiency IMG EXT
								: (ext === "PNG" || ext === "BMP" || ext === "TIFF" || ext === "TIF" || ext === "HEIF" || ext === "HEIC" || ext === "SVG") ? "green"
								: extColor;
							
							// Scale font size relative to thumbnail size
							const fontScale = 0.08;              // tweak if needed
							const fontSize = Math.max(8, currentThumbSize * fontScale);
							
							// Margin from edges, also scaled
							const marginScale = 0.03;            // tweak if needed
							const margin = currentThumbSize * marginScale;
							
							ctx.fillStyle = extColor;
							ctx.font = `bold ${fontSize}px sans-serif`;
							ctx.textBaseline = "top";
							ctx.textAlign = "left";
							
							// Text position
							const tx = x + margin;
							const ty = y + margin;
							
							// Optional: small shadow to improve clarity
							ctx.shadowColor = "rgba(0,0,0,0.7)";
							ctx.shadowBlur = fontSize * 0.25;
							ctx.shadowOffsetX = 0;
							ctx.shadowOffsetY = 0;
							
							ctx.fillText(ext, tx, ty);
							
							// Reset shadow
							ctx.shadowBlur = 0;
						}
						
						if (p.isVideo) {
							
							const isGif = ext && ext === "GIF";
							
							ctx.fillStyle = "#ffffff"; // white fill
							ctx.strokeStyle = isGif ? "#e98f00" : "#003e73"; // border -> orange for gifs and blue for videos
							ctx.lineJoin = "round"; // smoother corners
							
							// Triangle scale based on thumbnail size
							const triSize = currentThumbSize * 0.1;
							const halfH = triSize * 0.9;               // vertical half-height
							const rightW = triSize * 1.1;              // right point width
							const cx = x + currentThumbSize / 2;
							const cy = y + currentThumbSize / 2;
							
							ctx.beginPath();
							ctx.moveTo(cx - triSize,  cy - halfH);   // top-left
							ctx.lineTo(cx + rightW,   cy);           // right-center
							ctx.lineTo(cx - triSize,  cy + halfH);   // bottom-left
							ctx.closePath();
							
							ctx.lineWidth = triSize * 0.25; // proportional border width
							
							ctx.fill();
							ctx.stroke();
						}
                    } catch {}
                } else { // non-active/selected thumbnails that are NOT YET available
                    ctx.fillStyle = "#2d2d30";
                    ctx.fillRect(x + 2, y + 2, currentThumbSize - 4, currentThumbSize - 4);
					if (p.isVideo) { // tiny icon indicating video (triangle)
						// Triangle scale based on thumbnail size
						const triSize = currentThumbSize * 0.1;
						const halfH = triSize * 0.9;               // vertical half-height
						const rightW = triSize * 1.1;              // right point width
						const cx = x + currentThumbSize / 2;
						const cy = y + currentThumbSize / 2;
						
						ctx.fillStyle = "#a0a0a0";
						ctx.beginPath();
						ctx.moveTo(cx - triSize,  cy - halfH);   // top-left
						ctx.lineTo(cx + rightW,   cy);           // right-center
						ctx.lineTo(cx - triSize,  cy + halfH);   // bottom-left
						ctx.closePath();
						ctx.fill();
					}
                }
            }
			
            // highlight if selected
            if (i === selectedIndex) {
                ctx.lineWidth = 6;
                ctx.strokeStyle = "#00fc00";
                ctx.strokeRect(x, y, currentThumbSize, currentThumbSize);
            }
        });
		
		// 💡 Must restore the context state after the drawing is done
		ctx.restore(); 
		
        // draw a simple scrollbar indicator
        const totalRows = Math.ceil(posts.length / thumbsPerRow);
        const totalHeight = totalRows * rowHeight;
        if (totalHeight > thumbAreaH) {
            const handleH = Math.max(20, (thumbAreaH * (thumbAreaH / totalHeight)));
            const maxOffset = totalHeight - thumbAreaH;
            const handleY = thumbAreaY + (scrollOffset / maxOffset) * (thumbAreaH - handleH);
            ctx.fillStyle = "#3e3e42";
            roundRect(ctx, this.size[0] - SCROLLBAR_WIDTH, thumbAreaY + 4, SCROLLBAR_WIDTH - 2, thumbAreaH - 8, 6);
            ctx.fill();
            ctx.fillStyle = "#6f6f6f";
            roundRect(ctx, this.size[0] - SCROLLBAR_WIDTH + 2, handleY, SCROLLBAR_WIDTH - 6, handleH, 4);
            ctx.fill();
        }
    };

    // mouse & scroll handling for thumbnail area
    node.onMouseDown = function(event) {
		if (this.flags.collapsed) return false;
		const posY = TOP_PADDING;
		const localY = event.canvasY - this.pos[1];
		const localX = event.canvasX - this.pos[0];
		if (localY < posY) return false;
	
		// Determine clicked thumbnail
		const currentThumbSize = thumbnailSizeWidget ? thumbnailSizeWidget.value : THUMB_SIZE; 
		const thumbAreaH = this.size[1] - TOP_PADDING - 10;
		const width = this.size[0] - SCROLLBAR_WIDTH - 10;
		const thumbsPerRow = Math.max(1, Math.floor(width / (currentThumbSize + THUMB_PADDING)));
		const rowHeight = currentThumbSize + THUMB_PADDING;
		const relY = localY - posY + scrollOffset - THUMB_PADDING;
		const relX = localX - 8;
		const row = Math.floor(relY / rowHeight);
		const col = Math.floor(relX / (currentThumbSize + THUMB_PADDING));
		const idx = row * thumbsPerRow + col;
	
		// Calculate scrollbar geometry
		const SCROLLBAR_X = this.size[0] - SCROLLBAR_WIDTH;
		const totalRows = Math.ceil(posts.length / thumbsPerRow);
		const totalHeight = totalRows * rowHeight;
		const maxOffset = Math.max(0, totalHeight - thumbAreaH);
	
		// 💡 FIX 1: Check if click is on the scrollbar FIRST
		if (localX >= SCROLLBAR_X && totalHeight > thumbAreaH) {
			// Clicked in the scrollbar area, only proceed if scrolling is needed
			
			// 💡 Set up for SCROLLBAR DRAG
			isDragging = true;
			dragStartY = event.canvasY;
			scrollStart = scrollOffset;
			this.isScrollingHandle = true; // Flag for scrollbar drag vs content drag (though content drag is now disabled)
	
			// Calculate handle position for track jumping logic
			const handleH = Math.max(20, (thumbAreaH * (thumbAreaH / totalHeight)));
			const handleY = posY + 4 + (scrollOffset / maxOffset) * (thumbAreaH - 8 - handleH);
			
			// Check if the click was on the scrollbar track (not the handle)
			if (localY < handleY || localY > handleY + handleH) {
				// Clicked on the track, so jump the scroll handle
				const trackY = localY - (this.pos[1] + posY); // Y relative to track start
				
				// Calculate new scroll offset based on where the click was
				const trackStart = 4;
				const trackEnd = thumbAreaH - 4;
				const clickRatio = (localY - (this.pos[1] + trackStart + posY)) / (trackEnd - trackStart - handleH);
				
				const newOffset = clickRatio * maxOffset;
				scrollOffset = Math.max(0, Math.min(maxOffset, newOffset));
				
				// Re-center the handle under the cursor for a better feel
				scrollOffset = Math.max(0, Math.min(maxOffset, scrollOffset));
			}
	
			node.setDirtyCanvas(true);
			return true; // EAT THE EVENT - prevents LiteGraph from processing it further
		}
	
		// 💡 Original Thumbnail Selection logic (only runs if scrollbar wasn't clicked)
		if (idx >= 0 && idx < posts.length) {
			// Check if the click was within a thumbnail's bounding box
			const thumbX = 8 + col * (currentThumbSize + THUMB_PADDING);
			const thumbY_scrolled = posY + THUMB_PADDING + row * rowHeight - scrollOffset;
			
			// Check bounds (crude, but better than nothing)
			if (localX >= thumbX && localX <= thumbX + currentThumbSize &&
				localY >= thumbY_scrolled && localY <= thumbY_scrolled + currentThumbSize) {
				
				// Set selection
				selectedIndex = idx;
				if (selectedUrlWidget) selectedUrlWidget.value = posts[idx].file_url;
				if (selectedIMGTagsWidget) selectedIMGTagsWidget.value = posts[idx].tags;
				
				// If active video is this thumbnail, check if click hits controls (play/pause or scrub)
                if (videoMode && idx === activeVideoIndex && activeVideo && videoReady) {
                    const controlHit = handleVideoControlsClick(localX, localY, thumbX, thumbY_scrolled, currentThumbSize);
                    if (controlHit) {
                        node.setDirtyCanvas(true);
                        return true;
                    }
                }
				
				// If clicked thumbnail is not the active video, open it (or select)
				if (posts[idx].isVideo) {
					if (!videoMode || idx !== activeVideoIndex) {
						node.setDirtyCanvas(true);
						// Open video viewer for this thumbnail (async)
						openVideoForThumbnail(idx).catch(e => {
							console.error("[silver_fl_booru] openVideo error:", e);
						});
					} else {
						// If we clicked the thumbnail but it's already active and we didn't hit controls: toggle playback
						toggleVideoPlay();
					}
				} else {
					videoDuration = 0;
					videoRotation = 0;
					if (currentTimeMSWidget) currentTimeMSWidget.value = 0;
					prevTimestampMsForWidget = 0;
					activeVideoIndex = -1;
					videoMode = false;
				}
				node.setDirtyCanvas(true);
				
				return true;
			}
		}
		
		// 💡 FIX 2: If we reach here, it's a click in the empty thumbnail area, or outside everything. 
		// We shouldn't start dragging the content.
		return false;
    };
	
	
	node.onMouseMove = function(event) {
		const thumbSize = thumbnailSizeWidget ? thumbnailSizeWidget.value : DEFAULT_THUMB_SIZE;
		const rowHeight = thumbSize + THUMB_PADDING;
		const width = this.size[0] - SCROLLBAR_WIDTH - 10;
		const thumbsPerRow = Math.max(1, Math.floor(width / (thumbSize + THUMB_PADDING)));
		
		// --- NEW: Mouse Over Check ---
		const posY = TOP_PADDING;
        const localY = event.canvasY - this.pos[1];
        const localX = event.canvasX - this.pos[0];
        const relY = localY - posY + scrollOffset - THUMB_PADDING;
        const relX = localX - 8;
        const row = Math.floor(relY / rowHeight);
        const col = Math.floor(relX / (thumbSize + THUMB_PADDING));
        const idx = row * thumbsPerRow + col;
		// Check if mouse is within the rectangular area of the thumbnail grid
		if (idx >= 0 && idx < posts.length) {
			mouseOverThumbsGrid = true;
		} else {
			mouseOverThumbsGrid = false;
		}
		
        if (!isDragging) return false;
		
        const delta = event.canvasY - dragStartY;
        
        const thumbAreaH = this.size[1] - TOP_PADDING - 10;
        const totalRows = Math.ceil(posts.length / thumbsPerRow);
        const totalHeight = totalRows * rowHeight;
        const maxOffset = Math.max(0, totalHeight - thumbAreaH);

        const handleH = Math.max(20, (thumbAreaH * (thumbAreaH / totalHeight)));
        const scrollTrackMovement = thumbAreaH - 8 - handleH;
        const deltaRatio = delta / scrollTrackMovement;
        const newOffset = scrollStart + deltaRatio * maxOffset;
        scrollOffset = Math.max(0, Math.min(maxOffset, newOffset));
        node.setDirtyCanvas(true);
        return true;
    };
	
	
    node.onMouseUp = function(event) {
        isDragging = false;
		this.isScrollingHandle = false; 
        return false;
    };
	
	
	document.addEventListener('keydown', (e) => {
		if ((!nodeSelected && !mouseOverThumbsGrid && !isFullscreen) || activeIsGIF) { 
			return; 
		}
		// 1. Check if a video is active and the key press is NOT in an input field (e.g., a text box widget)
		// Checking activeElement prevents hotkeys from firing while a user is typing in ComfyUI widgets.
		const isTyping = document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA';
		if (activeVideo && !isTyping) {
			
			let shouldStep = false;
			let direction = '';
	
			if (e.key === 'ArrowRight') {
				direction = 'next';
				shouldStep = true;
			} else if (e.key === 'ArrowLeft') {
				direction = 'prev';
				shouldStep = true;
			}
	
			if (shouldStep) {
				// Prevent default browser behavior (like scrolling)
				e.preventDefault(); 
				stepVideoFrame(direction, e);
			}
		}
	});
	
    // Utility: round rect
    function roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
    }

    // perform search: call backend then load thumbnails
    async function performSearch() {
		
		activeVideoIndex = -1;
		videoMode = false;
		selectedIndex = -1;
		scrollOffset = 0;
		node.setDirtyCanvas(true);
		
        // read widget values
        const payload = {
            site: siteWidget ? siteWidget.value : "Gelbooru",
			AND_tags: AND_tagsWidget ? AND_tagsWidget.value : "",
            OR_tags: OR_tagsWidget ? OR_tagsWidget.value : "",
            exclude_tags: excludeWidget ? excludeWidget.value : "animated,",
            limit: limitWidget ? limitWidget.value : 20,
            page: pageWidget ? pageWidget.value : 1,
            Safe: safeWidget ? safeWidget.value : true,
            Questionable: questionableWidget ? questionableWidget.value : true,
            Explicit: explicitWidget ? explicitWidget.value : true,
			Order: orderWidget ? orderWidget.value : "Date",
			thumbnail_quality: thumbnailQualityWidget ? thumbnailQualityWidget.value : "Low",
            gelbooru_user_id: gelbooru_userWidget ? gelbooru_userWidget.value : "",
            gelbooru_api_key: gelbooru_apiWidget ? gelbooru_apiWidget.value : "",
			danbooru_user_id: danbooru_userWidget ? danbooru_userWidget.value : "",
            danbooru_api_key: danbooru_apiWidget ? danbooru_apiWidget.value : "",
			e621_user_id: e621_userWidget ? e621_userWidget.value : "",
            e621_api_key: e621_apiWidget ? e621_apiWidget.value : ""
        };

        try {
            const resp = await fetch("/silver_fl_booru/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await resp.json();
            if (data.error) {
                console.error("Booru search error:", data.error);
                posts = [];
                thumbnails = {};
                node.setDirtyCanvas(true);
                return;
            }
            posts = data.posts || [];
            thumbnails = {};
			
			const currentThumbSize = thumbnailSizeWidget ? thumbnailSizeWidget.value : THUMB_SIZE; 
			const thumbQuality = thumbnailQualityWidget ? thumbnailQualityWidget.value : "Low"; 
			
			// load thumbnail bitmaps (concurrently, but limited)
            // Use preview_url if provided by backend; backend returns preview_url in 'preview_url' but we used 'file_url' name.
            const loadPromises = posts.map(async (p, i) => {
                // prefer preview_url if exists
                const thumbUrl = p.preview_url || p.file_url || p.source || "";
                try {
                    const tResp = await fetch("/silver_fl_booru/thumb", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ url: thumbUrl, size: currentThumbSize - 4, thumbnail_quality: thumbQuality, site: p.site, user_id: p.user_id, api_key: p.api_key })
                    });
                    if (tResp.ok) {
                        const blob = await tResp.blob();
                        const bitmap = await createImageBitmap(blob);
                        thumbnails[i] = bitmap;
                        node.setDirtyCanvas(true);
                    } else {
                        // fallback: try to load file_url directly (CORS may block)
                        thumbnails[i] = null;
                        node.setDirtyCanvas(true);
                    }
                } catch (e) {
                    // ignore per-thumbnail errors
                    thumbnails[i] = null;
                }
            });

            // wait for all to start (not necessary to await all fully)
            Promise.all(loadPromises).catch(e => { /* ignore */ });

        } catch (e) {
            console.error("performSearch error:", e);
            posts = [];
            thumbnails = {};
            node.setDirtyCanvas(true);
        }
    }
	
	
	// VIDEO --------
	// Draw small video controls overlay inside the thumbnail
    function drawVideoControls(ctx, x, y, size) {
		
		// --- MAXIMIZE BUTTON (Top-Right Corner) ---
		const maxBtnSize = Math.floor(size * 0.1);
		const maxBtnX = x + size - maxBtnSize - 6;
		const maxBtnY = y + 6;
		ctx.save();
		ctx.strokeStyle = "#fff";
		ctx.lineWidth = 2;
		ctx.fillStyle = "rgba(0,0,0,0.55)";
		roundRect(ctx, maxBtnX, maxBtnY, maxBtnSize, maxBtnSize, 4);
		ctx.fill();
		// draw icon
		ctx.beginPath();
		if (!isFullscreen) {
			// maximize: draw a square outline
			ctx.strokeRect(maxBtnX + 4, maxBtnY + 4, maxBtnSize - 8, maxBtnSize - 8);
		} else {
			// minimize/close: draw X
			ctx.moveTo(maxBtnX + 4, maxBtnY + 4);
			ctx.lineTo(maxBtnX + maxBtnSize - 4, maxBtnY + maxBtnSize - 4);
			ctx.moveTo(maxBtnX + maxBtnSize - 4, maxBtnY + 4);
			ctx.lineTo(maxBtnX + 4, maxBtnY + maxBtnSize - 4);
		}
		ctx.stroke();
		ctx.restore();
		
		if (activeIsGIF) return;
		
        const barHeight = Math.max(22, Math.floor(size * 0.12));
        const barY = y + size - barHeight - 6;
        const padding = 6;
	
        // translucent background for controls
        ctx.fillStyle = "rgba(0,0,0,0.55)";
        roundRect(ctx, x + 4, barY, size - 8, barHeight, 6);
        ctx.fill();
	
        // play/pause button
        const btnX = x + 12;
        const btnY = barY + Math.floor((barHeight - 14) / 2);
        const btnSize = 14;
        ctx.fillStyle = "#ffffff";
	
        if (isVideoPlaying) {
            // pause icon (two bars)
            ctx.fillRect(btnX, btnY, 4, btnSize);
            ctx.fillRect(btnX + 8, btnY, 4, btnSize);
        } else {
            // play triangle
            ctx.beginPath();
            ctx.moveTo(btnX, btnY);
            ctx.lineTo(btnX + btnSize, btnY + btnSize / 2);
            ctx.lineTo(btnX, btnY + btnSize);
            ctx.closePath();
            ctx.fill();
        }
	
        // progress bar
        const progressX = btnX + btnSize + 10;
        const progressW = size - (progressX - x) - 18;
        const progressY = barY + Math.floor(barHeight / 2) - 4;
        ctx.fillStyle = "#444";
        roundRect(ctx, progressX, progressY, progressW, 8, 4);
        ctx.fill();
	
        let t = 0;
		let compensatedCurrentTime = 0;
        if (videoReady && activeVideo && videoDuration > 0) {
			compensatedCurrentTime = activeVideo.currentTime;
			t = compensatedCurrentTime / videoDuration;
        }
        const fillW = Math.max(2, Math.floor(progressW * t));
        ctx.fillStyle = "#00d400";
        roundRect(ctx, progressX, progressY, fillW, 8, 4);
        ctx.fill();
	
        // timestamp
        ctx.font = "11px Arial";
		ctx.fillStyle = "#ddd";
		ctx.textAlign = "right";
		const totalTime = formatTime(videoDuration);
		const currentTimeStr = videoReady && activeVideo ? formatTime(compensatedCurrentTime) : "0:00"; 
		ctx.fillText(`${currentTimeStr} / ${totalTime}`, x + size - 16, barY + barHeight / 2 + 4);
    }
	

    function formatTime(seconds) {
		if (!isFinite(seconds)) return "0:00";
		seconds = Math.max(0, Math.floor(seconds));
		const s = seconds % 60;
		const m = Math.floor(seconds / 60);
		return `${m}:${s.toString().padStart(2, "0")}`;
    }
	
	// Handle clicks in the video controls region. Returns true if the click was handled.
	function handleVideoControlsClick(localX, localY, thumbX, thumbY, thumbSize) {
		const barHeight = Math.max(22, Math.floor(thumbSize * 0.12));
		const barY = thumbY + thumbSize - barHeight - 6;
	
		// ----------------------------------------------------------
		// 1. FIRST: Maximize / Minimize button (top-right corner)
		// ----------------------------------------------------------
		const maxBtnSize = Math.floor(thumbSize * 0.1);
		const maxBtnX = thumbX + thumbSize - maxBtnSize - 6;
		const maxBtnY = thumbY + 6;
	
		if (
			localX >= maxBtnX &&
			localX <= maxBtnX + maxBtnSize &&
			localY >= maxBtnY &&
			localY <= maxBtnY + maxBtnSize
		) {
			// Toggle fullscreen mode
			if (!isFullscreen) enterFullscreen();
			else exitFullscreen();
			return true;
		}
	
		// ----------------------------------------------------------
		// 2. Play/Pause button
		// ----------------------------------------------------------
		const btnX = thumbX + 12;
		const btnY = barY + Math.floor((barHeight - 14) / 2);
		const btnSize = 14;
	
		if (
			localX >= btnX &&
			localX <= btnX + btnSize &&
			localY >= btnY &&
			localY <= btnY + btnSize
		) {
			toggleVideoPlay();
			return true;
		}
	
		// ----------------------------------------------------------
		// 3. Progress bar seek
		// ----------------------------------------------------------
		const progressX = btnX + btnSize + 10;
		const progressW = thumbSize - (progressX - thumbX) - 18;
		const progressY = barY + Math.floor(barHeight / 2) - 4;
	
		if (
			localX >= progressX &&
			localX <= progressX + progressW &&
			localY >= progressY &&
			localY <= progressY + 8
		) {
			const rel = (localX - progressX) / progressW;
			scrubVideoTo(rel);
			return true;
		}
	
		return false;
	}

    function toggleVideoPlay() {
        if (!activeVideo || !videoReady) return;
        if (isVideoPlaying) {
            activeVideo.pause();
            isVideoPlaying = false;
            cancelAnimationFrame(rafId);
            rafId = null;
            // update selected frame (timestamp)
            updatecurrentTimeMSWidget();
        } else {
            // start playing; ensure we have playing promise handling
            const playPromise = activeVideo.play();
            if (playPromise && playPromise.then) {
                playPromise.then(() => {
                    isVideoPlaying = true;
                    startVideoRaf();
                }).catch((e) => {
                    console.warn("video play prevented:", e);
                });
            } else {
                // older browsers
                isVideoPlaying = true;
                startVideoRaf();
            }
        }
    }

    function startVideoRaf() {
        if (rafId) cancelAnimationFrame(rafId);
        function loop() {
            // keep redrawing while playing
            node.setDirtyCanvas(true);
            updatecurrentTimeMSWidget(); // update widget to current timestamp frequently
            rafId = requestAnimationFrame(loop);
        }
        rafId = requestAnimationFrame(loop);
    }

    function scrubVideoTo(normalized) {
        if (!activeVideo || !videoReady || !isFinite(videoDuration)) return;
        const clamped = Math.max(0, Math.min(1, normalized));
        const newTime = clamped * videoDuration;
		
		try {
			activeVideo.currentTime = newTime;
		} catch (e) {
			console.warn("Standard seek failed:", e);
		}
		
        // ensure paused on scrub
        if (!isVideoPlaying) {
            updatecurrentTimeMSWidget();
            node.setDirtyCanvas(true);
        }
    }

    function updatecurrentTimeMSWidget() {
        if (!currentTimeMSWidget || activeIsGIF) return;
        if (activeVideo && videoReady && isFinite(activeVideo.currentTime)) {
            const ms = Math.floor( (activeVideo.currentTime) * 1000);
            // only write when changed
            if (ms !== prevTimestampMsForWidget) {
                currentTimeMSWidget.value = ms;
                prevTimestampMsForWidget = ms;
            }
        }
        if (selectedUrlWidget && activeVideoFileURL != null) {
            selectedUrlWidget.value = activeVideoFileURL;
        }
    }

    async function openVideoForThumbnail(i) {
		
		videoDuration = 0;
		videoRotation = 0;
		if (currentTimeMSWidget) currentTimeMSWidget.value = 0;
		prevTimestampMsForWidget = 0;
		
		// i is index within current visible posts array
		let duration_ms = 0;
		try {
			const resp = await fetch("/silver_fl_booru/videoprobe", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({file_url: posts[i].file_url})
			});
			const data = await resp.json();
			if (data.error) {
				console.log("openVideoForThumbnail error", data.error);
				return;
			}
			duration_ms = data.duration_ms || 0;
			videoRotation = data.rotation || 0;
		} catch { return; }
		
		// reset old viewer mode
		activeVideoIndex = -1;
		videoMode = false;
		// force thumbnail revert
		node.setDirtyCanvas(true);
		
		videoDuration = duration_ms / 1000;
			
		try {
			// If another video is active, clean it up
			if (activeVideo) {
				activeVideo = null;
				videoReady = false;
				isVideoPlaying = false;
				if (!activeIsGIF) {
					try {
						activeVideo.pause();
						activeVideo.src = "";
						activeVideo.removeAttribute("src");
						activeVideo.load();
					} catch {}
					try {
						if (activeVideo.src && activeVideo.src.startsWith("blob:")) {
							URL.revokeObjectURL(activeVideo.src);
						}
					} catch {}
					if (rafId) {
						cancelAnimationFrame(rafId);
						rafId = null;
					}
				}
			}
			
			await new Promise(requestAnimationFrame);
			
			activeVideoIndex = i;
			activeVideoFileURL = posts[i].file_url;
			videoMode = true;
			selectedIndex = i;
			if (selectedUrlWidget) selectedUrlWidget.value = posts[i].file_url;
			if (selectedIMGTagsWidget) selectedIMGTagsWidget.value = posts[i].tags;
			
			
			activeIsGIF = posts[i].file_url.endsWith(".gif")// || posts[i].file_url.endsWith(".gifv")
			const v = !activeIsGIF ? document.createElement("video") : document.createElement("img");
			
			
			
			// Create or reuse a container DIV for fullscreen
			videoContainer = null;
			if (!videoContainer) {
				videoContainer = document.createElement("div");
				videoContainer.style.position = "fixed";
				videoContainer.style.left = "0";
				videoContainer.style.top = "0";
				videoContainer.style.width = "100%";
				videoContainer.style.height = "100%";
				videoContainer.style.background = "black";
				videoContainer.style.display = "none";
				videoContainer.style.zIndex = "9999";
				videoContainer.style.justifyContent = "center";
				videoContainer.style.alignItems = "center";
				videoContainer.style.flexDirection = "column";
				videoContainer.style.display = "flex";
				
				if (!activeIsGIF) {
					
					// Play/Pause on left click
					videoContainer.addEventListener("click", (e) => {
						// prevent clicking controls from pausing
						const isVideoClick = e.target === v;
						if (isVideoClick) {
							toggleVideoPlay();
						}
					});
					
					// Controls for fullscreen mode
					const fsControls = document.createElement("div");
					fsControls.style.position = "absolute";
					fsControls.style.bottom = "10px";
					fsControls.style.left = "10px";
					fsControls.style.right = "10px";
					fsControls.style.display = "flex";
					fsControls.style.alignItems = "center";
					fsControls.style.justifyContent = "space-between";
					fsControls.style.background = "rgba(0,0,0,0.6)";
					fsControls.style.padding = "5px";
					fsControls.style.zIndex = "10000";
					videoContainer.appendChild(fsControls);
					
					const playPauseFsBtn = document.createElement("button");
					playPauseFsBtn.textContent = "Play";
					fsControls.appendChild(playPauseFsBtn);
					
					progressFs = document.createElement("input");
					progressFs.type = "range";
					progressFs.min = "0";
					progressFs.max = "100";
					progressFs.value = "0";
					progressFs.style.flex = "1";
					fsControls.appendChild(progressFs);
					
					// START: Time Display Element
					fullScreentimeDisplay = document.createElement("span");
					fullScreentimeDisplay.style.color = "#ddd";
					fullScreentimeDisplay.style.font = "11px Arial";
					fullScreentimeDisplay.style.margin = "0 10px"; // Add some spacing
					updateFullScreenTimeDisplay();
					fsControls.appendChild(fullScreentimeDisplay);
					// END: Time Display Element
					
					const closeFsBtn = document.createElement("button");
					closeFsBtn.textContent = "✕"; // close symbol
					fsControls.appendChild(closeFsBtn);
					
					// Play / Pause logic
					playPauseFsBtn.addEventListener("click", () => {
						toggleVideoPlay();
						if (v.paused) {
							playPauseFsBtn.textContent = "Pause";
						} else {
							playPauseFsBtn.textContent = "Play";
						}
					});
					
					// Scrub logic
					progressFs.addEventListener("input", (e) => {
						const norm = e.target.value / 100;
						const newTime = norm * videoDuration;
						v.currentTime = newTime;
					});
					
					// Close / minimize logic
					closeFsBtn.addEventListener("click", () => {
						exitFullscreen();
					});
					
					// Update progress in loop
					v.addEventListener("timeupdate", () => {
						const percent = (v.currentTime / videoDuration) * 100;
						progressFs.value = percent;
						updateFullScreenTimeDisplay();
					});
				}
				
			}
			
			let src_url = "/silver_fl_booru/videostream?file_url=" + activeVideoFileURL;
			
			if (!activeIsGIF) {
				const source = document.createElement("source");
				source.src = src_url;
				v.appendChild(source);
				
				v.muted = audioMutedWidget.value;
				v.volume = audioMutedWidget.value ? 0 : 1.0;
				v.preload = "auto";
				v.playsInline = true;
				v.crossOrigin = "anonymous";
				v.preload = "metadata";
				
				v.style.maxWidth = "100%";
				v.style.maxHeight = "100%";
				v.style.objectFit = "contain"; // Crucial for scaling tall videos
				
				// event listeners
				videoReady = false;
				v.addEventListener("loadedmetadata", () => {
					videoReady = true;
					// set initial selected frame (0)
					updatecurrentTimeMSWidget();
					node.setDirtyCanvas(true);
				});
				v.addEventListener("error", (e) => {
					console.error("[silver_fl_booru] video element error:", e);
					videoReady = false;
					node.setDirtyCanvas(true);
				});
			
			} else {
				v.src = src_url;
			}
	
			// Set active video
			activeVideo = v;
			
			if (isFullscreen && videoContainer) {
				// 1. Remove the old video element (if any)
				while (videoContainer.querySelector('video')) {
					videoContainer.removeChild(videoContainer.querySelector('video'));
				}
				
				// 2. Remove img for gifs
				while (videoContainer.querySelector('img')) {
					videoContainer.removeChild(videoContainer.querySelector('img'));
				}
				
				// 3. Insert the new active video element at the top
				videoContainer.prepend(activeVideo);
			}
	
			// Prepare a single poster frame by seeking to start (some containers don't allow drawImage until play/resume)
			try { v.currentTime = 0; } catch {}
			//try { await v.pause(); } catch {}
	
			videoReady = activeIsGIF || !!(v.readyState >= 2);
			node.setDirtyCanvas(true);
			
		} catch (e) {
			console.error("[silver_fl_booru] openVideoForThumbnail exception:", e);
			videoMode = false;
			activeVideo = null;
			activeVideoIndex = -1;
			activeVideoFileURL = null;
			activeIsGIF = false;
			videoReady = false;
			isVideoPlaying = false;
			node.setDirtyCanvas(true);
		}
    }
	
	
	function stepVideoFrame(direction, event) {
		// 1. Check if a video element is currently active and ready
		if (!activeVideo || !videoReady) {
			return;
		}
		
		let multiplier = 1;
		// We assume the caller passes the keyboard event object as the second argument
		if (event) {
			if (event.shiftKey) { multiplier = 5; }
			if (event.ctrlKey) { multiplier = 0.25; }
		}
	
		// 2. Calculate the change in time
		const step = (direction === 'next' ? FRAME_STEP : -FRAME_STEP) * multiplier;
		const newTime = Math.max(0, Math.min(activeVideo.currentTime + step, videoDuration));
		
		// 3. Apply the new time and pause
		activeVideo.currentTime = newTime;
		activeVideo.pause();
		
		// 4. Update your ComfyUI widget/display
		updatecurrentTimeMSWidget();
		node.setDirtyCanvas(true);
	}
	
	function enterFullscreen() {
		if (!videoContainer || !videoReady) return;
		
		if (isVideoPlaying) {
			activeVideo.pause();
			isVideoPlaying = false;
		}
		
		// Remove old video if present (safety cleanup)
		while (videoContainer.querySelector('video')) {
			videoContainer.removeChild(videoContainer.querySelector('video'));
		}
		
		// Append the currently active video to the container
		videoContainer.prepend(activeVideo);
		
		// This ensures the video container is visible and attached to the body
		if (!videoContainer.parentNode) {
			document.body.appendChild(videoContainer);
		}
		
		videoContainer.style.display = "flex";
		videoContainer.requestFullscreen?.().catch((err) => {
			console.warn("Failed to request fullscreen:", err);
		});
		isFullscreen = true;
	}
	
	function exitFullscreen() {
		if (isVideoPlaying) {
			activeVideo.pause();
			isVideoPlaying = false;
		}
		
		if (document.fullscreenElement) {
			document.exitFullscreen().catch((err) => {
			console.warn("Exit fullscreen failed:", err);
			});
		}
		// Remove the video element from the fullscreen container
		if (videoContainer && activeVideo && videoContainer.contains(activeVideo)) {
			videoContainer.removeChild(activeVideo);
		}
		
		if (videoContainer) {
			videoContainer.style.display = "none";
		}
		isFullscreen = false;
	}
	
	document.addEventListener("fullscreenchange", () => {
		if (!document.fullscreenElement) {
			// we have exited fullscreen
			exitFullscreen();
		}
	});
	// --------------
	
	
	function updateNodeSize() {
        const width = Math.max(MIN_WIDTH, node.size[0]);
        const height = Math.max(MIN_HEIGHT, node.size[1]);
        node.size[0] = width;
        node.size[1] = height;
    }
	
    node.onResize = function() {
        updateNodeSize();
        this.setDirtyCanvas(true);
    };
	
    updateNodeSize();
    // initial search on node creation? we won't auto-search; user must press Search
    node.setDirtyCanvas(true);
}


