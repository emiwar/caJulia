
import QtQuick 2.0
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.0
import org.julialang 1.0
//import QtQuick.Controls.Material 2.12

ApplicationWindow {
  title: "Calcium Trace Extraction with Julia"
  width: 512
  height: 512
  visible: true
  //applicationName: "CaJulia"
  //Material.theme: Material.Dark
  Connections {
        target: timer
        onTimeout: Julia.checkworkerstatus(observables)
    }

  menuBar: MenuBar {
    Menu {
        title: qsTr("File")
        Action { 
            text: qsTr("Open video") 
            onTriggered: fileDialog.visible = true;
        }
        Action { 
            text: qsTr("Ping worker") 
            onTriggered: Julia.pingworker();
        }
        Action { 
            text: qsTr("Quit") 
            onTriggered: Qt.quit();
        }
    }
    Menu {
        title: qsTr("Process")
        Action { 
            text: qsTr("Calculate init image") 
            onTriggered: Julia.calcinitframe();
        }
        Action { 
            text: qsTr("Find footprints") 
            onTriggered: Julia.initfootprints();
        }
        Action { 
            text: qsTr("Initiate backgrounds") 
            onTriggered: Julia.initbackgrounds();
        }
        Action { 
            text: qsTr("Update traces") 
            onTriggered: Julia.updatetraces();
        }
        Action { 
            text: qsTr("Update footprints") 
            onTriggered: Julia.updatefootprints();
        }
        Action { 
            text: qsTr("Merge cells") 
            onTriggered: Julia.mergecells();
        }
    }
  }
  footer: RowLayout {
        spacing: 20
        Layout.fillWidth: true
        Text {
            id: footerStatus
            text: observables.status_text
        }
        RowLayout {
            spacing: 6
            ProgressBar {
                Layout.fillWidth: true
                id: footerStatusProgressBar
                value: observables.status_progress
            }
            Text {
                id: footerStatusProgressBarLabel
                text: (100*observables.status_progress).toPrecision(3)+"%"
            }
            visible: observables.status_progress >= 0.0
        }
    }
  ColumnLayout {
    spacing: 6
    anchors.centerIn: parent
    anchors.fill: parent
    anchors.leftMargin: 5
    anchors.topMargin: 5
    anchors.rightMargin: 5
    anchors.bottomMargin: 5
    GridLayout {
        Layout.preferredHeight: 50
        Layout.fillWidth: true
        columns: 4
        rows: 2
        JuliaCanvas {
            id: viewport1
            paintFunction: paint_cfunction1
            Layout.fillWidth: true
            Layout.preferredWidth: 300
            Layout.minimumWidth: 50
            Layout.minimumHeight: width
            Layout.maximumHeight: width
            Layout.row: 0
            Layout.column: 0
            MouseArea {
                anchors.fill: parent
                property int lastX
                property int lastY
                onWheel: (mouse) => {
                    Julia.zoomscroll(mouse.x/width, mouse.y/height, mouse.angleDelta.y/8, observables)
                }
                onPressed: (mouse) => {
                    lastX = mouse.x
                    lastY = mouse.y
                }
                onPositionChanged: (mouse) => {
                    Julia.pandrag((mouse.x - lastX)/width, (mouse.y - lastY)/height, observables)
                    lastX = mouse.x
                    lastY = mouse.y
                }
            }
        }
        RangeSlider {
            Layout.fillWidth: true
            id: viewport1ContrastSlider
            from: 1
            to: 512
            second.value: 512
            first.onMoved: {
                observables.cmin1 = first.value
            }
            second.onMoved: {
                observables.cmax1 = second.value
            }
            Layout.row: 1
            Layout.column: 0
        }
        JuliaCanvas {
            id: viewport2
            paintFunction: paint_cfunction2
            Layout.fillWidth: true
            Layout.preferredWidth: 300
            Layout.minimumWidth: 50
            Layout.minimumHeight: width
            Layout.maximumHeight: width
            Layout.row: 0
            Layout.column: 1
            MouseArea {
                anchors.fill: parent
                property int lastX
                property int lastY
                onWheel: (mouse) => {
                    Julia.zoomscroll(mouse.x/width, mouse.y/height, mouse.angleDelta.y/8, observables)
                }
                onPressed: (mouse) => {
                    lastX = mouse.x
                    lastY = mouse.y
                }
                onPositionChanged: (mouse) => {
                    Julia.pandrag((mouse.x - lastX)/width, (mouse.y - lastY)/height, observables)
                    lastX = mouse.x
                    lastY = mouse.y
                }
            }
        }
        RangeSlider {
            Layout.fillWidth: true
            id: viewport2ContrastSlider
            from: 1
            to: 512
            second.value: 512
            first.onMoved: {
                observables.cmin2 = first.value
            }
            second.onMoved: {
                observables.cmax2 = second.value
            }
            Layout.row: 1
            Layout.column: 1
        }
        JuliaCanvas {
            id: viewport3
            paintFunction: paint_cfunction3
            Layout.fillWidth: true
            Layout.preferredWidth: 300
            Layout.minimumWidth: 50
            Layout.minimumHeight: width
            Layout.maximumHeight: width
            Layout.row: 0
            Layout.column: 2
            MouseArea {
                anchors.fill: parent
                property int lastX
                property int lastY
                onWheel: (mouse) => {
                    Julia.zoomscroll(mouse.x/width, mouse.y/height, mouse.angleDelta.y/8, observables)
                }
                onPressed: (mouse) => {
                    lastX = mouse.x
                    lastY = mouse.y
                }
                onPositionChanged: (mouse) => {
                    Julia.pandrag((mouse.x - lastX)/width, (mouse.y - lastY)/height, observables)
                    lastX = mouse.x
                    lastY = mouse.y
                }
            }
        }
        RangeSlider {
            Layout.fillWidth: true
            id: viewport3ContrastSlider
            from: -8
            to: 4
            first.value: observables.cmin3
            second.value: observables.cmax3
            first.onMoved: {
                observables.cmin3 = first.value
            }
            second.onMoved: {
                observables.cmax3 = second.value
            }
            Layout.row: 1
            Layout.column: 2
        }
        
        JuliaCanvas {
            id: viewport4
            paintFunction: paint_cfunction4
            Layout.fillWidth: true
            Layout.preferredWidth: 300
            Layout.minimumWidth: 50
            Layout.minimumHeight: width
            Layout.maximumHeight: width
            MouseArea {
                anchors.fill: parent
                property int lastX
                property int lastY
                onReleased: (mouse) => {
                    Julia.footprintclick(mouse.x/width, mouse.y/height, observables)
                }
                onWheel: (mouse) => {
                    Julia.zoomscroll(mouse.x/width, mouse.y/height, mouse.angleDelta.y/8, observables)
                }
                onPressed: (mouse) => {
                    lastX = mouse.x
                    lastY = mouse.y
                }
                onPositionChanged: (mouse) => {
                    Julia.pandrag((mouse.x - lastX)/width, (mouse.y - lastY)/height, observables)
                    lastX = mouse.x
                    lastY = mouse.y
                }
            }
            Layout.row: 0
            Layout.column: 3
        }
    }
    //Rectangle {
    //    Layout.fillWidth: true
    //    Layout.fillHeight: true
    //    color: "white"
    Canvas {
        id: traceCanvas
        Layout.fillWidth: true
        Layout.fillHeight: true
        contextType: "2d"
        antialiasing: true
        onPaint: {
            var ctx = getContext( "2d" );
            ctx.save();
            ctx.clearRect( 0, 0, width, height);
            for(var trace_id=0; trace_id<2; trace_id += 1) {
                var y = observables["trace"+("RCS"[trace_id])]
                var x_max = y.length-1, y_max=1.5
                if(trace_id == 1) {
                    ctx.strokeStyle = observables["traceCol"]
                } else {
                    ctx.strokeStyle = "#AAAAAA"
                }
                ctx.beginPath()
                ctx.moveTo(0, (1-(y[0]+.4)/y_max)*height)
                for(var i=1; i<y.length; i+=1) {
                    ctx.lineTo(i*width/x_max, (1-((y[i]+.4))/y_max)*height)
                }
                ctx.stroke()
            }
            ctx.restore()
            //context.strokeStyle = Qt.rgba(.4,.6,.8)
            //context.path = myPath
            //context.stroke()
        }
    }

    //}
    RowLayout {
        Slider {
            id: time_slider
            value: 0.0
            //from: 1
            //to: observables.n_frames
            //step: 1
            //snapMode: Slider.SnapAlways
            Layout.fillWidth: true
            Layout.fillHeight: false
            Layout.minimumWidth: 100
            //Layout.minimumHeight: 100
            onValueChanged: {
                observables.frame_n_float = value;
                //viewport1.update();
                //viewport2.update();
            }
        }
        Text {
            id: timeSliderLabel
            text: ""+observables.frame_n
        }
        Button {
            id: stepBackButton
            text: "🡠"
            Layout.preferredWidth: height
            onClicked: {
                observables.frame_n_float -= 1.0/observables.n_frames
            }
        }
        Button {
            id: stepForwardButton
            text: "🡢"
            Layout.preferredWidth: height
            onClicked: {
                observables.frame_n_float += 1.0/observables.n_frames
            }
        }
        Button {
            id: playButton
            text: "⏵"
            Layout.preferredWidth: height
        }
    }
  }

  JuliaSignals {
    signal updateDisplay(var disp_id)
    onUpdateDisplay: {
        if (disp_id == 1) viewport1.update()
        else if (disp_id == 2) viewport2.update()
        else if (disp_id == 3) viewport3.update()
        else if (disp_id == 4) viewport4.update()
        else if (disp_id == 5) traceCanvas.requestPaint()
    }
  }

  FileDialog {
    id: fileDialog
    title: "Open video"
    folder: shortcuts.home
    selectMultiple: false
    selectExisting: true
    selectFolder: false
    onAccepted: {
        Julia.openvideo(fileDialog.fileUrl)
    }
    //onRejected: {
    //    console.log("Canceled")
    //}
    //Component.onCompleted: visible = true
  }

  Component.onCompleted: {
    viewport1.update()
    viewport2.update()
    viewport3.update()
    viewport4.update()
    timer.start();
  }
}