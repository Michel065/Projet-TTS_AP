#include "model/Tool/graphs_tool.h"

void create_graphs_loss(std::vector<float>& histo_loss,int height, int width){
    if(histo_loss.empty()) return;

    float min = histo_loss[0];
    float max = histo_loss[0];
    for(float v : histo_loss){
        if(v < min) min = v;
        if(v > max) max = v;
    }

    std::vector<float> data;

    if((int)histo_loss.size() <= width){
        data = histo_loss;
    } else {
        int step = histo_loss.size() / (width-1);

        for(int i = 0; i < width-1; i++){
            float sum = 0.0f;
            int count = 0;

            for(int j = i * step; j < (i+1) * step && j < (int)histo_loss.size(); j++){
                sum += histo_loss[j];
                count++;
            }

            data.push_back(sum / count);
        }
    }

    int size = data.size();
    // ----------- affichage -----------
    std::string label = "loss";
    int label_pos = height / 2 - label.size() / 2;

    for(int h = height; h >= 0; h--){
        
        // affichage du label vertical
        if(h >= label_pos && h < label_pos + (int)label.size()){
            std::cout << label[label_pos + label.size() - 1 - h];
        } else {
            std::cout << " ";
        }

        std::cout << "|";

        for(int i = 0; i < size; i++){
            float norm = (data[i] - min) / (max - min + 1e-8f);
            int level = (int)(norm * height);

            if(level == h)
                std::cout << "*";
            else
                std::cout << " ";
        }

        std::cout << "\n";
    }

    // ----------- axe horizontal -----------
    int misize = size / 2;    
    int add_=0;
    std::cout << " +";
    for(int i = 0; i < size; i++){
        if(add_ == 0 && i >= misize - 3){
            std::cout << "epochs";
            i += 5;
            add_ = 1;
        } else {
            std::cout << "-";
        }
    }
    std::cout << "\n";


    // ----------- stats -----------
    Print("First loss : ", histo_loss.front());
    Print("Last loss  : ", histo_loss.back());
    Print("Loss min : ", min);
    Print("Loss max : ", max);
    Print("Delta (max - min) : ", max - min);
}