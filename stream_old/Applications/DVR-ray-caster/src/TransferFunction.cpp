#include <TransferFunction.hpp>

#include <GcCore/libMath/Vector.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui/imgui.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace tdns
{
namespace graphics
{   
    TransferFunction::TransferFunction(int samplesCount):
        samplesCount(samplesCount)
    {
        samples.resize(samplesCount);

        controlPoint hp  = {0.00f, 0.00f, -1, {0.f, 0.f, 0.f}};
        controlPoint cp0 = {0.20f, 0.10f,  0, {0.f, 0.f, 1.f}};
        controlPoint cp1 = {0.35f, 0.70f,  1, {0.f, 1.f, 1.f}};
        controlPoint cp2 = {0.45f, 1.00f,  2, {0.f, 1.f, 0.f}};
        controlPoint cp3 = {0.70f, 1.00f,  3, {1.f, 1.f, 0.f}};
        controlPoint cp4 = {1.00f, 1.00f,  4, {1.f, 0.f, 0.f}};

        hoveredPoint = hp;
        curvesControlPoints.push_back(cp0);
        curvesControlPoints.push_back(cp1);
        curvesControlPoints.push_back(cp2);
        curvesControlPoints.push_back(cp3);
        curvesControlPoints.push_back(cp4);

        regenerateSamples(); // Regenerate the samples
    }

    float TransferFunction::lerp(float a, float b, float t) const
    {
        return a + (b - a) * t;
    }

    float TransferFunction::invLerp(float a, float b, float v) const
    {
         return (v - a) / (b - a);
    }

    ImVec2 TransferFunction::lerp(const ImVec2& a, const ImVec2& b, const ImVec2& t)
    {
        return ImVec2(
            lerp(a.x, b.x, t.x),
            lerp(a.y, b.y, t.y)
        );
    }

    ImVec2 TransferFunction::invLerp(const ImVec2& a, const ImVec2& b, const ImVec2& v)
    {
        return ImVec2(
            invLerp(a.x, b.x, v.x),
            invLerp(a.y, b.y, v.y)
        );
    }

    void TransferFunction::clamp01(ImVec2& position)
    {
        position.x = std::clamp(position.x, 0.f, 1.f);
        position.y = std::clamp(position.y, 0.f, 1.f);
    }

    float TransferFunction::sqrDistance(const ImVec2& a, const ImVec2& b)
    {
        ImVec2 ab(
            a.x - b.x,
            a.y - b.y
        );
        return ab.x * ab.x + ab.y * ab.y;
    }


    ImVec2 TransferFunction::getPointPosition(const controlPoint& values, const ImRect& bb)
    {
        return lerp(
            bb.Min, bb.Max,
            ImVec2(
                values.value,
                1.f - values.alpha
            )
        );
    }

    std::tuple<float, float> TransferFunction::getPointValues(const ImVec2& position, const ImRect& bb)
    {
        auto values = invLerp(bb.Min, bb.Max, position);
        clamp01(values);
        return std::make_tuple(values.x, 1.f - values.y);
    }

    bool TransferFunction::isValidPoint(const controlPoint point) {
        return point.value >= 0 && point.alpha >= 0 && point.value <= 1 && point.alpha <= 1;
    }

    void TransferFunction::regenerateSamples()
    {
        for(int i = 0; i < samplesCount; i++)
            samples[i] = getSample(static_cast<float>(i)/static_cast<float>(samplesCount));
    }

    glm::vec4 TransferFunction::getSample(float xValue) const
    {
        glm::vec4 sample(0.f, 0.f, 0.f, 0.f);

        if(curvesControlPoints.empty())
            return sample;

        // Find the previous and next control point
        auto followingCtrlPIt = std::find_if(curvesControlPoints.begin(),
                                            curvesControlPoints.end(),
                                            [xValue](const auto& values){return values.value > xValue;});           
        int pos = std::distance(curvesControlPoints.begin(), followingCtrlPIt);

        controlPoint previousValues = (pos == 0) ? 
            controlPoint({0.f, 0.f, -1, curvesControlPoints[pos].rgb}) : curvesControlPoints[pos - 1];

        controlPoint nextValues = followingCtrlPIt == curvesControlPoints.end() ? 
            controlPoint{1.f, 0.f, -1, curvesControlPoints.back().rgb} : *followingCtrlPIt;

        // Compute the sample value by interpolating between the previous and next point
        const float t = invLerp(previousValues.value, nextValues.value, xValue);

        sample = glm::vec4(
            lerp(previousValues.rgb.r, nextValues.rgb.r, t),
            lerp(previousValues.rgb.g, nextValues.rgb.g, t),
            lerp(previousValues.rgb.b, nextValues.rgb.b, t),
            lerp(previousValues.alpha, nextValues.alpha, t)
        );

        return sample;
    }

    int TransferFunction::draw(const ImVec2& size)
    {
        drawHeader(); // Draw the header

        // Draw the editor
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        ImGuiWindow* window = ImGui::GetCurrentWindow();

        // Prepare canvas
        const float availableWidth = ImGui::GetContentRegionAvail().x;
        // Set the size of the canvas
        const ImVec2 canvas(
            size.x > 0 ? size.x : availableWidth - ImGui::GetStyle().FramePadding.x * 10,
            size.y > 0 ? size.y : (size.x / 2.f)
        );

        // Prepare the bounding box of the canvas
        const ImRect bb(
            ImVec2(
                window->DC.CursorPos.x + ImGui::GetStyle().FramePadding.x, 
                window->DC.CursorPos.y + ImGui::GetStyle().FramePadding.y*3),
            ImVec2(
                window->DC.CursorPos.x + canvas.x,
                window->DC.CursorPos.y + canvas.y
            )
        );

        ImGui::ItemSize(bb);
        if (!ImGui::ItemAdd(bb, 0)) {
            return 0;
        }

        const ImGuiID id = window->GetID("Transfer Function");
        ImGui::ItemHoverable(bb, id, ImGuiItemFlags_None);

        // Draw histogram
        const ImColor histogramColor(.65f, .65f, .65f, .3f);
        for (size_t i = 0; i < histogram.size(); i++){
            const auto& values = histogram[i];
            const int count = std::get<0>(values);
            const float start = std::get<1>(values);
            const float end = std::get<2>(values);
            const float height =  count / static_cast<float>(dataCount);
            const ImVec2 startPos = ImVec2(bb.Min.x + start * canvas.x, bb.Max.y);
            const ImVec2 endPos = ImVec2(bb.Min.x + end * canvas.x, bb.Max.y - height * canvas.y);
            drawList->AddRectFilled(startPos, endPos, histogramColor);   
        }

        // Draw background lines
        float splits = std::floor(canvas.x / 25.f);
        
        for (int i = 0; i <= splits; ++i) {
            const float pos = (canvas.x / splits) * static_cast<float>(i);
            drawList->AddLine(
                ImVec2(bb.Min.x + pos, bb.Min.y),
                ImVec2(bb.Min.x + pos, bb.Max.y),
                ImGui::GetColorU32(ImGuiCol_TextDisabled)
            );
        }

        for (int i = 0; i <= 4; ++i) {
            const float pos = (canvas.y / 4.f) * static_cast<float>(i);
            drawList->AddLine(
                ImVec2(bb.Min.x, bb.Min.y + pos),
                ImVec2(bb.Max.x, bb.Min.y + pos),
                ImGui::GetColorU32(ImGuiCol_TextDisabled)
            );
        }
    
        // Draw samples
        bool changed = false;

        // Handle user interactions
        handleHoveredPoint(bb);
        changed |= handlerInputs(bb);

        // Draw all the curves
        drawCurve(drawList, bb);

        if(changed)
            regenerateSamples();

        return changed;
    }

    void TransferFunction::drawHeader()
    {
        // Draw the helper
        ImGui::SameLine(ImGui::GetWindowWidth() - 30);
        // ImGui::SetNextItemWidth(30);
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::BulletText("Double left click to add point");
            ImGui::BulletText("Left click drag to move point");
            ImGui::BulletText("Right click to edit point");
            ImGui::EndTooltip();
        }
    }

    void TransferFunction::drawCurve(ImDrawList* drawList, const ImRect& bb)
    {
        auto& controlPoints = curvesControlPoints;

        // Draw the curve
        ImColor lineColor = ImColor(0.5f, 0.5f, 0.5f, 1.f);

        if(controlPoints.empty()){
            drawList->AddLine(getPointPosition({0, 0, -1, {0.f, 0.f, 0.f}}, bb),
                            getPointPosition({1, 1, -1, {0.f, 0.f, 0.f}}, bb),
                            lineColor,
                            3);
        }else{
            for(int i = -1; i < (int)controlPoints.size(); i++){
                controlPoint first, second;
                if (i == -1) 
                    first = {0.f, 0.f, -1, controlPoints[0].rgb};
                else 
                    first = controlPoints[i];
                if (i == (int)controlPoints.size() - 1)
                    second = {1.f, 0.f, -1, controlPoints[i].rgb};
                else
                    second = controlPoints[i + 1];
                
                lineColor = ImColor(
                    lerp(first.rgb.r, second.rgb.r, 0.5f),
                    lerp(first.rgb.g, second.rgb.g, 0.5f),
                    lerp(first.rgb.b, second.rgb.b, 0.5f),
                    1.f
                );
                
                drawList->AddLine(
                    getPointPosition(first, bb),
                    getPointPosition(second, bb),
                    lineColor,
                    3
                );
            }
        }

        // Draw the control points
        for (size_t i = 0; i < controlPoints.size(); ++i) {
            controlPoint& ctrlPoint = controlPoints[i];
            const auto position = getPointPosition(ctrlPoint, bb);
            ImColor color = ImColor(ctrlPoint.rgb.r, ctrlPoint.rgb.g, ctrlPoint.rgb.b, 1.f);
            drawList->AddCircleFilled(position, PointsRadius, color);
            if (isValidPoint(hoveredPoint) && (size_t)hoveredPoint.index == i){
                drawList->AddCircle(
                    position, PointsRadius,
                    ImColor(0.f, 0.f, 0.f, 1.f)
                );

                ImGui::SetTooltip(
                    "x=%1.4f\ny=%1.4f",
                    ctrlPoint.value,
                    ctrlPoint.alpha
                );
            }
        }
    }

    void TransferFunction::handleHoveredPoint(const ImRect& bb)
    {
        const ImVec2 mousePos = ImGui::GetIO().MousePos;

        controlPoint nearestPoint = {-1.f, -1.f, -1, {0.f, 0.f, 0.f}};
        auto nearestPointSqrDist = std::numeric_limits<float>::infinity();

        // Find the nearest point to the mouse
        for (size_t j = 0; j < curvesControlPoints.size(); ++j) {
            const auto guiPos = getPointPosition(curvesControlPoints[j], bb);
            const auto sqrDist = sqrDistance(guiPos, mousePos);

            if (sqrDist < nearestPointSqrDist) {
                nearestPointSqrDist = sqrDist;
                nearestPoint = curvesControlPoints[j];
                nearestPoint.index = j;
            }
        }

        // If the mouse is over a point, set the hovered point
        if (nearestPointSqrDist <= SqrPointSelectionRadius) {
            hoveredPoint = nearestPoint;
        } else {
        // Otherwise, set the hovered point to invalid
            hoveredPoint = {-1.f, -1.f, -1, {0.f, 0.f, 0.f}};
        }
    }

    bool TransferFunction::handlerInputs(const ImRect& bb)
    {
        const auto& mousePosition = ImGui::GetIO().MousePos;

        auto handleDragging = [this, mousePosition, bb]() -> bool {
            if (!ImGui::IsPopupOpen("edit_point_popup")
                && ImGui::IsMouseDragging(0, 0.1)) {
                
                // If the current dragging point is not valid, set the dragging point to the hovered point
                if (!isValidPoint(draggedPoint)) {
                    if (!isValidPoint(hoveredPoint)) {
                        return true;
                    } else{
                        draggedPoint = hoveredPoint;
                    }
                }

                // Get the position of the dragged point in the vector
                auto index = draggedPoint.index;

                auto point = getPointValues(mousePosition, bb);
                // Erase the point from the vector
                curvesControlPoints.erase(
                    curvesControlPoints.begin()
                    + index
                );
                // Find the next point to the mouse position
                auto pos = std::find_if(
                    curvesControlPoints.begin(),
                    curvesControlPoints.end(),
                    [point](const auto& p) {
                        return p.value >= std::get<0>(point);
                    }
                );
                // Set the new point information and insert it in the vector
                auto newIndex = std::distance(
                    curvesControlPoints.begin(),
                    pos
                );
                controlPoint insertPoint = {
                    std::get<0>(point),
                    std::get<1>(point),
                    (int)newIndex,
                    draggedPoint.rgb
                };
                curvesControlPoints.insert(pos, insertPoint);
                draggedPoint = insertPoint;

                return true;
            } else {
                draggedPoint = {-1.f, -1.f, -1, {0.f, 0.f, 0.f}};
                return false;
            }
        };

        auto handleInsertPoint = [this, mousePosition, bb]() -> bool {
            if (!ImGui::IsPopupOpen("edit_point_popup")
            && ImGui::IsMouseDoubleClicked(0)
            && bb.Contains(mousePosition)) {
                auto point = getPointValues(mousePosition, bb);
                glm::vec3 rgb;
                // Find the next point to the mouse position
                auto pos = std::find_if(
                    curvesControlPoints.begin(),
                    curvesControlPoints.end(),
                    [point, &rgb](const auto& p) {
                        if (p.value > std::get<0>(point)) {
                            rgb = p.rgb;
                            return true;
                        } 
                        return false;               
                    }
                );
                auto newIndex = std::distance(
                    curvesControlPoints.begin(),
                    pos
                );
                // Update position for other points
                for (size_t i = (size_t)newIndex; i < curvesControlPoints.size(); ++i) {
                    curvesControlPoints[i].index++;
                }
                // Create the new point and insert it in the vector
                controlPoint insertPoint = {
                    std::get<0>(point),
                    std::get<1>(point),
                    (int)newIndex,
                    rgb
                };
                curvesControlPoints.insert(pos, insertPoint);
                return true;
            }
            return false;
        };

        auto handleEditPoint = [this]() -> bool {
            // If hovered point is valid, open the edit point popup
            if (ImGui::IsMouseClicked(1) && isValidPoint(hoveredPoint)) {
                editedPoint = hoveredPoint;
                ImGui::OpenPopup("edit_point_popup");
            } else if (!ImGui::IsPopupOpen("edit_point_popup")) {
                editedPoint = {-1.f, -1.f, -1, {0.f, 0.f, 0.f}};
            }

            bool edited = false;
            std::tuple<float, float> point = std::make_tuple(editedPoint.value, editedPoint.alpha);
            if (ImGui::BeginPopup("edit_point_popup")) {
                edited |= ImGui::ColorEdit3("RGB##edit_point_popup", glm::value_ptr(editedPoint.rgb));
                if (ImGui::Button("Delete##edit_point_popup")) {
                    curvesControlPoints.erase(
                        curvesControlPoints.begin() + editedPoint.index
                    );
                    edited = true;
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }
            if (edited){
                auto pos = std::find_if(
                    curvesControlPoints.begin(),
                    curvesControlPoints.end(),
                    [point](const auto& p) {
                        return p.value >= std::get<0>(point);
                    }
                );
                *pos = editedPoint;
            }
            return edited;
        };

        return handleDragging() || handleInsertPoint() || handleEditPoint();
    }
} // namespace graphics
} // namespace tdns