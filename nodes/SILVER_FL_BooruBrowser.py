import re
import requests
import io
import json
import os
from PIL import Image, ImageOps
import torch
import numpy as np
from aiohttp import web
import boto3
import comfy
from server import PromptServer


USER_AGENT = "ComfyUI-BooruBrowser/1.0"


def loadImageFromUrl(url):
    if url.startswith("data:image/"):
        i = Image.open(io.BytesIO(base64.b64decode(url.split(",")[1])))
    elif url.startswith("s3://"):
        s3 = boto3.client('s3')
        bucket, key = url.split("s3://")[1].split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        i = Image.open(io.BytesIO(obj['Body'].read()))
    else:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            raise Exception(response.text)

        i = Image.open(io.BytesIO(response.content))

    i = ImageOps.exif_transpose(i)

    if i.mode != "RGBA":
        i = i.convert("RGBA")

    # recreate image to fix weird RGB image
    alpha = i.split()[-1]
    image = Image.new("RGB", i.size, (0, 0, 0))
    image.paste(i, mask=alpha)

    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if "A" in i.getbands():
        mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

    return (image, mask)




# --- Node class ---
class SILVER_FL_BooruBrowser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "site": (["Gelbooru","Danbooru","E621"], {}),
                "AND_tags": ("STRING", {"default": "", "multiline": False}),
                "OR_tags": ("STRING", {"default": "", "multiline": False}),
                "exclude_tags": ("STRING", {"default": "animated,", "multiline": False}),
                "limit": ("INT", {"default": 20, "min": 1, "max": 100}),
                "page": ("INT", {"default": 1, "min": 1}),
                "Safe": ("BOOLEAN", {"default": True}),
                "Questionable": ("BOOLEAN", {"default": True}),
                "Explicit": ("BOOLEAN", {"default": True}),
                "Order": (["Date","Score"], {}),
                "thumbnail_size": ("INT", {"default": 240, "min": 80, "max": 1024}),
                "gelbooru_user_id": ("STRING", {"default": ""}),
                "gelbooru_api_key": ("STRING", {"default": ""}),
                "danbooru_user_id": ("STRING", {"default": ""}),
                "danbooru_api_key": ("STRING", {"default": ""}),
                "e621_user_id": ("STRING", {"default": ""}),
                "e621_api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                # This hidden widget is used by the JS UI to set selected image URL
                "selected_url": ("STRING", {"default": ""}),
                "selected_img_tags": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE","STRING")
    RETURN_NAMES = ("image","imgtags")
    FUNCTION = "browse_booru"
    CATEGORY = "Booru"
    DESCRIPTION = """
Quickly retrieve images from Gelbooru/Danbooru/E621 without leaving ComfyUI.

Notes: 
    - Does not work with videos/animations (TIP: ensure you have 'animated' in exclude_tags)
    - 'AND_tags' , 'OR_tags' and 'exclude_tags' are single line comma separated booru tags
    - user_id for Danbooru/E621 is actually your login id
    - Searching on Danbooru is mostly useless unless you have a Gold account (you are limited to 2 tags)
"""

    def browse_booru(self, site, AND_tags, OR_tags, exclude_tags, limit, page, Safe, Questionable, Explicit, Order, thumbnail_size,
        gelbooru_user_id, gelbooru_api_key, danbooru_user_id, danbooru_api_key, e621_user_id, e621_api_key, 
        selected_url="", selected_img_tags=""):
        
        if not selected_url:
            return (torch.zeros(1,1,1,3), "")
            
        try:
            img, mask = loadImageFromUrl(selected_url)
            img_tags = ", ".join([tag.strip() for tag in selected_img_tags.lower().replace(',', ' ').split(' ') if tag.strip() != ''])
            return (img, img_tags)
        except Exception as e:
            print("[SILVER_FL_BooruBrowser] error loading selected_url:", e)
            return (torch.zeros(1,1,1,3), "")

    @classmethod
    def IS_CHANGED(cls, site, AND_tags, OR_tags, exclude_tags, limit, page, Safe, Questionable, Explicit, Order, thumbnail_size,
        gelbooru_user_id, gelbooru_api_key, danbooru_user_id, danbooru_api_key, e621_user_id, e621_api_key, 
        selected_url="", selected_img_tags=""):
        # Node is considered changed when a thumbnail selection modifies selected_url
        return selected_url

    @classmethod
    def VALIDATE_INPUTS(cls, site, AND_tags, OR_tags, exclude_tags, limit, page, Safe, Questionable, Explicit, Order, thumbnail_size,
        gelbooru_user_id, gelbooru_api_key, danbooru_user_id, danbooru_api_key, e621_user_id, e621_api_key, 
        selected_url="", selected_img_tags=""):
        
        s = ""
        if page < 1:
            s = "[SILVER_FL_BooruBrowser] Page must be >= 1"
        if limit < 1 or limit > 100:
            s = "[SILVER_FL_BooruBrowser] Limit must be between 1 and 100"
        if site == "Gelbooru" and (gelbooru_user_id == "" or gelbooru_api_key == ""):
            s = "[SILVER_FL_BooruBrowser] Gelbooru requires both 'gelbooru_user_id' and 'gelbooru_api_key'"
        
        if s != "":
            print(s)
            return s
        
        return True


# --- HTTP endpoints for the UI (search + thumb proxy) ---
@PromptServer.instance.routes.post("/silver_fl_booru/search")
async def api_booru_search(request):
    """
    Request JSON body:
    {
      "site": "Gelbooru" or "Danbooru" or "E621",
      "AND_tags": "tag1, tag2",
      "OR_tags": "tag1, tag2",
      "exclude_tags": "animated,",
      "limit": int,
      "page": int,
      "Safe": bool,
      "Questionable": bool,
      "Explicit": bool,
      "Order": "Date" or "Score",
      "gelbooru_user_id": "",
      "gelbooru_api_key": "",
      "danbooru_user_id": "",
      "danbooru_api_key": "",
      "e621_user_id": "",
      "e621_api_key": ""
    }
    Returns: JSON { "posts": [ {id, file_url, preview_url, tags, width, height, source} ... ] }
    """
    try:
        data = await request.json()
        
        site = data.get("site", "Gelbooru")
        AND_tags = data.get("AND_tags", "")
        OR_tags = data.get("OR_tags", "")
        exclude_tag = data.get("exclude_tags", "")
        limit = int(data.get("limit", 20))
        page = int(data.get("page", 1))
        Safe = bool(data.get("Safe", True))
        Questionable = bool(data.get("Questionable", True))
        Explicit = bool(data.get("Explicit", True))
        Order = data.get("Order", "Date")
        
        gelbooru_user_id = data.get("gelbooru_user_id", "")
        gelbooru_api_key = data.get("gelbooru_api_key", "")
        danbooru_user_id = data.get("danbooru_user_id", "")
        danbooru_api_key = data.get("danbooru_api_key", "")
        e621_user_id = data.get("e621_user_id", "")
        e621_api_key = data.get("e621_api_key", "")
        
        # fix page by site if necessary
        if site == "Gelbooru":
            page -= 1 # Gelbooru API expects the fist page to be 0 but this node's minimum page value is 1
        
        # Normalize tags
        def normalize_tags(tag_str):
            tags = tag_str.rstrip(',').rstrip(' ')
            tags = tags.split(',')
            tags = [item.strip().replace(' ', '_').replace('\\', '') for item in tags]
            return [item for item in tags if item]
        
        
        AND_tags = normalize_tags(AND_tags)
        OR_tags = normalize_tags(OR_tags)
        exclude_tags = normalize_tags(exclude_tag)
        
        # Build tag query depending on site
        tag_query = []
        if site == "Gelbooru":
            if AND_tags:
                tag_query.append('+'.join(AND_tags))
            if OR_tags:
                if len(OR_tags) > 1:
                    tag_query.append('{' + ' ~ '.join(OR_tags) + '}')
                else:
                    tag_query.append(OR_tags[0])
            if exclude_tags:
                tag_query.append('+'.join('-' + t for t in exclude_tags))
        elif site == "E621":
            tag_query.extend(AND_tags)
            if OR_tags:
                # Each OR tag must be prefixed with "~"
                tag_query.extend([f"~{t}" for t in OR_tags])
            if exclude_tags:
                tag_query.extend(["-" + t for t in exclude_tags])
        elif site == "Danbooru":
            # Danbooru uses space-separated tags, with `-` prefix for exclusion
            tag_query.extend(AND_tags)
            if OR_tags:
                if len(OR_tags) > 1:
                    # Danbooru supports “(a or b)” syntax with ~?
                    # Actually Danbooru uses `a~b` for OR (tilde), so we convert OR_tags accordingly
                    tag_query.append("(" + "~".join(OR_tags) + ")")
                else:
                    tag_query.append(OR_tags[0])
            if exclude_tags:
                tag_query.extend(["-" + t for t in exclude_tags])
        
        
        # Rating filtering
        if site == "Gelbooru":
            if not Safe:
                tag_query.append("-rating:general")
            if not Questionable:
                tag_query.append("-rating:questionable -rating:sensitive")
            if not Explicit:
                tag_query.append("-rating:explicit")
        elif site == "E621":
            rating_filters = []
            if not Safe:
                rating_filters.append("-rating:s")
            if not Questionable:
                rating_filters.append("-rating:q")
            if not Explicit:
                rating_filters.append("-rating:e")
            tag_query.extend(rating_filters)
        elif site == "Danbooru":
            if not Safe:
                tag_query.append("-rating:s")
            if not Questionable:
                tag_query.append("-rating:q")
            if not Explicit:
                tag_query.append("-rating:e")
        
        # Apply ordering
        if Order == "Score":
            if site == "Gelbooru":
                tag_query.append("sort:score:desc")
            elif site == "Danbooru":
                tag_query.append("order:score_desc")
            elif site == "E621":
                tag_query.append("order:score_desc")
        
        # Base URL
        if site == "Gelbooru":
            base_url = "https://gelbooru.com/index.php"
        elif site == "E621":
            base_url = "https://e621.net/posts.json"
        elif site == "Danbooru":
            base_url = "https://danbooru.donmai.us/posts.json"
        
        user_id = (
            gelbooru_user_id if site == "Gelbooru" else
            danbooru_user_id if site == "Danbooru" else
            e621_user_id
        )
        api_key = (
            gelbooru_api_key if site == "Gelbooru" else
            danbooru_api_key if site == "Danbooru" else
            e621_api_key
        )
        
        
        # Build request
        params = {}
        headers = {"User-Agent": USER_AGENT}
        url = base_url
        
        if site == "Gelbooru":
            query = (
                f"page=dapi&s=post&q=index&tags="
                f"{'+'.join(tag_query)}"
                f"&limit={limit}&pid={page}&json=1&api_key={api_key}&user_id={user_id}"
            )
            url = f"{base_url}?{query}"
            url = re.sub(r"\++", "+", url)
        else:
            # E621 and Danbooru both use JSON endpoints and accept params
            params["limit"] = limit
            params["page"] = page
            params["tags"] = " ".join(tag_query)
            if user_id and api_key:
                params["login"] = user_id
                params["api_key"] = api_key
        
        resp = requests.get(url, params=(params if site != "Gelbooru" else None), headers=headers, timeout=10)
        resp.raise_for_status()
        
        posts = []
        if site == "Gelbooru":
            posts = resp.json().get("post", [])
        elif site == "E621":
            posts = resp.json().get("posts", [])
        elif site == "Danbooru":
            posts = resp.json()  # Danbooru returns an array of post objects   
        
        #print(f"[SILVER_FL_BooruBrowser] url: {url}")
        #print(f"[SILVER_FL_BooruBrowser] params: {params}")
        
        out_posts = []
        for post in posts:
            id = "" # optional
            file_url = "" # required
            preview_url = "" # required
            tags = "" # required - must be a string with tags separated by spaces
            source = "" # optional
            
            if site == "Gelbooru":
                id = post.get("id", "")
                source = post.get("source", "")
                file_url = post.get("file_url", "") or source # when file_url is not available use source instead
                preview_url = post.get("preview_url", "") or post.get("sample_url", "") or file_url # as a last resort: set it to file_url
                tags = post.get("tags", "")
            
            elif site == "E621":
                id = str(post.get("id", ""))
                source = post.get("sources", [None])[0] if post.get("sources") else ""
                file_url = post.get("file", {}).get("url", "") or source
                preview_url = post.get("preview", {}).get("url", "") or file_url
                # Tags in E621 are grouped by category (general, species, etc.)
                tag_groups = post.get("tags", {})
                all_tags = []
                for group_tags in tag_groups.values():
                    all_tags.extend(group_tags)
                tags = " ".join(all_tags)
            
            elif site == "Danbooru":
                id = str(post.get("id", ""))
                source = post.get("source", "")
                # Danbooru returns fields like file_url, large_file_url, preview_file_url
                file_url = post.get("large_file_url") or post.get("file_url") or source
                preview_url = post.get("preview_file_url") # non-Gold users won't get a 'preview_file_url' for gold-exclusive files - so we take advantage of that and don't fallback to file_url here
                tags = post.get("tag_string", "")  # Danbooru’s tag string
            
            #print(f"[SILVER_FL_BooruBrowser] preview_url: {preview_url}")
            #print(f"[SILVER_FL_BooruBrowser] file_url: {file_url}")
            
            if file_url and preview_url and tags:
                out_posts.append({
                    "id": id,
                    "file_url": file_url,
                    "preview_url": preview_url,
                    "tags": tags,
                    "source": source,
                    "site": site,
                    "user_id": user_id,
                    "api_key": api_key
                })
        
        
        return web.json_response({"posts": out_posts})
    except Exception as e:
        print("[SILVER_FL_BooruBrowser] /search error:", e)
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/silver_fl_booru/thumb")
async def api_booru_thumb(request):
    """
    Request JSON body: { "url": "<image url>", "size": 80 }
    Returns PNG thumbnail bytes (image/png)
    """
    try:
        data = await request.json()
        
        url = data.get("url", "")
        size = int(data.get("size", 240))
        
        # in case its necessary to parse the url or add cookies with these
        site = data.get("site", "")    
        user_id = data.get("user_id", "")
        api_key = data.get("api_key", "")
        
        if not url:
            return web.json_response({"error": "No url provided"}, status=400)
        
        # Fetch image (timeout small)
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, timeout=8, stream=True, headers=headers)
        resp.raise_for_status()
        
        #img = Image.open(io.BytesIO(resp.content))
        img = Image.open(resp.raw)
        
        img = ImageOps.exif_transpose(img)
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        
        return web.Response(body=buf.read(), content_type='image/jpeg')
    except Exception as e:
        print("[SILVER_FL_BooruBrowser] /thumb error:", e)
        return web.json_response({"error": str(e)}, status=500)





NODE_CLASS_MAPPINGS = {
    "SILVER_FL_BooruBrowser": SILVER_FL_BooruBrowser,
}



# Provide a display name mapping so it looks nice in the node list (optional)
NODE_DISPLAY_NAME_MAPPINGS = {
    "SILVER_FL_BooruBrowser": "[Silver] Booru Browser",
}


