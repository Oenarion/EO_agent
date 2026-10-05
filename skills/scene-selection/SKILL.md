---
name: scene-selection
description: How to choose among Sentinel-2 scenes for a place and a period. Use it when the user asks which scene is best, which one to pick or use, or to rank or compare scenes. Do not use it for a plain search or for the details of one scene.
---

# Choosing a scene

The catalogue cannot say which scene is "the best": it depends on what the user wants to do, and the cloud figure it gives is computed over the whole scene (about 113 km wide), not over the user's area. So never name a winner. Give a short list of candidates, with reasons, and say what is not known.

## Procedure

1. Note the use if the user gave one (crops, city, water, forest, burn scars...). If not, answer anyway and say that you did not assume a use.
2. Search the place and the period in the usual way. Use the cloud limit only if the user gave one. The results come sorted by cloud cover, clearest first.
3. Take the 3 most promising scenes (lowest cloud cover) and call get_scene_details for each one, to read the sun elevation.
4. Rank them in this order of criteria:
   - the lowest cloud cover in the catalogue;
   - the date closest to the date or season the user wants;
   - when cloud covers are within 5 points of each other, the higher sun elevation (less shadow, better light).
5. Answer with at most 3 scenes. For each: the id, the date, the cloud cover, the sun elevation, and one line of reason.
6. Finish with this limit, in the user's language: the cloud figure is for the whole scene, so a scene can be cloudier or clearer over the requested area. Suggest opening the preview link before using it.

## Words

Say "the clearest by the catalogue figure" or "a good candidate". Never say "the best scene".
