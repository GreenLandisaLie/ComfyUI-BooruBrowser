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
	const thumbnailSizeWidget = node.widgets.find(w => w.name === "thumbnail_size");
	const thumbnailQualityWidget = node.widgets.find(w => w.name === "thumbnail_quality");
	
    const gelbooru_userWidget = node.widgets.find(w => w.name === "gelbooru_user_id");
    const gelbooru_apiWidget = node.widgets.find(w => w.name === "gelbooru_api_key");
	const danbooru_userWidget = node.widgets.find(w => w.name === "danbooru_user_id");
    const danbooru_apiWidget = node.widgets.find(w => w.name === "danbooru_api_key");
	const e621_userWidget = node.widgets.find(w => w.name === "e621_user_id");
    const e621_apiWidget = node.widgets.find(w => w.name === "e621_api_key");
	
    // hidden selected url widget (JS sets this when user clicks a thumbnail)
    const selectedUrlWidget = node.widgets.find(w => w.name === "selected_url");
	// hidden selected img tags widget (JS sets this when user clicks a thumbnail)
    const selectedIMGTagsWidget = node.widgets.find(w => w.name === "selected_img_tags");

    // Make sure selected_url exists; hide it from the UI
    if (selectedUrlWidget) selectedUrlWidget.hidden = true;
	// Make sure selected_img_tags exists; hide it from the UI
    if (selectedIMGTagsWidget) selectedIMGTagsWidget.hidden = true;

    // UI sizing constants
	const MIN_WIDTH = 500;
    const MIN_HEIGHT = 875;
    const TOP_PADDING = 610;
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

            // draw image if ready
            const bm = thumbnails[i];
            if (bm) {
                try {
                    ctx.drawImage(bm, x + 2, y + 2, currentThumbSize - 4, currentThumbSize - 4);
                } catch (e) {
                    // drawing failed; ignore
                }
            } else {
                // placeholder loading box
                ctx.fillStyle = "#2d2d30";
                ctx.fillRect(x + 2, y + 2, currentThumbSize - 4, currentThumbSize - 4);
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
				
				// Update selection and set hidden widget value to the full file_url
				selectedIndex = idx;
				const item = posts[idx];
				if (selectedUrlWidget) {
					selectedUrlWidget.value = item.file_url;
				}
				if (selectedIMGTagsWidget) {
					selectedIMGTagsWidget.value = item.tags;
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
		// isDragging should only be true if we started dragging the scrollbar handle
		if (!isDragging) return false;
		
		const delta = event.canvasY - dragStartY;
		
		const currentThumbSize = thumbnailSizeWidget ? thumbnailSizeWidget.value : THUMB_SIZE; 
		const thumbAreaH = this.size[1] - TOP_PADDING - 10;
		const width = this.size[0] - SCROLLBAR_WIDTH - 10;
		const thumbsPerRow = Math.max(1, Math.floor(width / (currentThumbSize + THUMB_PADDING)));
		const rowHeight = currentThumbSize + THUMB_PADDING;
		const totalRows = Math.ceil(posts.length / thumbsPerRow);
		const totalHeight = totalRows * rowHeight;
		
		const maxOffset = Math.max(0, totalHeight - thumbAreaH); // Max scrollable distance
		
		// 💡 FIX 3: Scrollbar handle dragging logic
		// Track height available for handle movement (total track height - handle height)
		const handleH = Math.max(20, (thumbAreaH * (thumbAreaH / totalHeight)));
		const scrollTrackMovement = thumbAreaH - 8 - handleH; 
		
		// Calculate the ratio of cursor movement (delta) to handle movement space
		const deltaRatio = delta / scrollTrackMovement;
		
		// New scroll offset should be based on initial scroll + movement mapped to total scroll range
		const newOffset = scrollStart + (deltaRatio * maxOffset);
	
		// Clamp the new scroll offset
		scrollOffset = Math.max(0, Math.min(maxOffset, newOffset));
		
		// Note: Since you mentioned the scrollbar logic was previously backwards, 
		// if the scrolling feels inverted (dragging down scrolls up), change `newOffset` calculation to:
		// const newOffset = scrollStart - (deltaRatio * maxOffset); 
		// (But the current fix should be correct for mapping handle movement)
	
		node.setDirtyCanvas(true);
		return true;
    };

	

    node.onMouseUp = function(event) {
        isDragging = false;
		this.isScrollingHandle = false; 
        return false;
    };

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
            selectedIndex = -1;
            scrollOffset = 0;
            node.setDirtyCanvas(true);
			
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

