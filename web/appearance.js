import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "SILVER-Nodes.appearance", // Extension name
    async nodeCreated(node) {
        if (node.comfyClass.startsWith("SILVER_FL_BooruBrowser")) {
            node.color = "#4F0074";
            node.bgcolor = "#003e73";
        }
    }
});