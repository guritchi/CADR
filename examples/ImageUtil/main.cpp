#define STB_IMAGE_IMPLEMENTATION
#include "../../3rdParty/stb/stb_image.h"

#include <iostream>

int main(int argc, char** argv) {
    if (argc <= 2) {
        std::cerr << "Missing argument\n";
        return 1;
    }

    int width = 0;
    int height = 0;
    int channels = 0;

    float *data = stbi_loadf(argv[1], &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "Failed to load image: " << argv[1] << "\n";
        return 1;
    }
    int targetChannel = std::atoi(argv[2]);
    if (targetChannel < 0 || targetChannel >= channels) {
        std::cerr << "Invalid channel " << targetChannel << " out of " << channels << "\n";
    }
    // std::cout << "Loaded " << argv[1] << ": " << width << "x" << height << ", " << channels << " channels\n";
    if (argc >= 4) {
        int downsample = atoi(argv[3]);
        if (downsample <= 0) {
            std::cerr << "Invalid downsample value\n";
        }
        int w = width / downsample;
        int h = height / downsample;
        float area = downsample * downsample;

        std::cout << "float BUFFER[" << h << "][" << w <<  "] = {\n";
        for (int y = 0; y < height; y += downsample) {
            std::cout << "  {";
            for (int x = 0; x < width; x += downsample) {
                float value = 0;
                for (int py = 0; py < downsample; ++py) {
                    for (int px = 0; px < downsample; ++px) {
                        float* pixel = data + (((y + py) * width + (x + px)) * channels);
                        value += pixel[targetChannel];
                    }
                }
                std::cout << (value / area) << ",";
            }
            std::cout << "  },\n";
        }
        std::cout << "};\n";
        return 0;
    }

    std::cout << "float BUFFER[" << height << "][" << width <<  "] = {\n";
    for (int y = 0; y < height; ++y) {
        std::cout << "  {";
        for (int x = 0; x < width; ++x) {
            float value = 0;
            float* pixel = data + ((y * width + x) * channels);
            std::cout << pixel[targetChannel] << ",";
        }
        std::cout << "  },\n";
    }
    std::cout << "};\n";

    stbi_image_free(data);
    return 0;
}
