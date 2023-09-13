import QtQuick 2.0
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.0
import org.julialang 1.0
import QtQuick.Controls.Material 2.15

ApplicationWindow {
  title: "Calcium Trace Extraction with Julia"
  width: 1024
  height: 1024*(3.0/4.0)
  visible: true
  //applicationName: "CaJulia"
  Material.theme: Material.Dark
  //Material.accent: Material.Purple
  Connections {
        target: timer
        onTimeout: Julia.checkworkerstatus(observables)
    }

  menuBar: MenuBar {
    Menu {
        id: fileMenu
        title: qsTr("File")

        Action {
            text: qsTr("Open video") 
            onTriggered: openVideoDialog.visible = true;
        }
        Action { 
            text: qsTr("Save result") 
            onTriggered: saveResultDialog.visible = true;
        }
        Action { 
            text: qsTr("Open behavior video") 
            onTriggered: openBehaviorDialog.visible = true;
        }
        Action { 
            text: qsTr("Open result") 
            onTriggered: openResultsDialog.visible = true;
        }
        Action { 
            text: qsTr("Ping worker") 
            onTriggered: Julia.pingworker();
        }
        Action { 
            text: qsTr("Reset worker") 
            onTriggered: Julia.resetworker();
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
        Action { 
            text: qsTr("Motion correct") 
            onTriggered: Julia.motioncorrect();
        }
        Action { 
            text: qsTr("Subtract min") 
            onTriggered: Julia.subtractmin();
        }
        Action { 
            text: qsTr("Delete selected cell") 
            onTriggered: Julia.deleteselectedcell(observables);
        }
        Action { 
            text: qsTr("Clear filter") 
            onTriggered: Julia.clearfilter();
        }
    }
    Menu {
        title: qsTr("Settings")
        Action { 
            id: sameContrastCheckbox
            text: qsTr("Same contrast")
            checkable: true
            checked: false
            onTriggered: {
                if (sameContrastCheckbox.checked) {
                    observables.cmin2 = observables.cmin1
                    observables.cmax2 = observables.cmax1
                }
            }
        }
    }
  }
  footer: RowLayout {
        spacing: 20
        Layout.fillWidth: true
        Text {
            id: footerStatus
            text: observables.status_text
            color: "#FFFFFF"
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
                color: "#FFFFFF"
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
        columns: 5
        rows: 3
        Text {
            Layout.row: 0
            Layout.column: 0
            Layout.preferredWidth: 150
            text: "<b>Raw video</b>"
            color: "#CCCCCC"
        }
        Text {
            Layout.row: 0
            Layout.column: 1
            Layout.preferredWidth: 150
            text: "<b>Reconstructed video</b>"
            color: "#CCCCCC"
        }
        Text {
            Layout.row: 0
            Layout.column: 2
            Layout.preferredWidth: 150
            text: "<b>Negentropy summary</b>"
            color: "#CCCCCC"
        }
        Text {
            Layout.row: 0
            Layout.column: 3
            Layout.preferredWidth: 150
            text: "<b>Footprints</b>"
            color: "#CCCCCC"
        }
        Text {
            Layout.row: 0
            Layout.column: 4
            Layout.preferredWidth: 150
            text: "<b>Behavior</b>"
            color: "#CCCCCC"
        }
        JuliaCanvas {
            id: viewport1
            paintFunction: paint_cfunction1
            Layout.fillWidth: true
            Layout.preferredWidth: 300
            Layout.minimumWidth: 50
            Layout.minimumHeight: width
            Layout.maximumHeight: width
            Layout.row: 1
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
            from: observables.crangemin//0
            to: observables.crangemax//2048
            second.value: 32//256
            first.onMoved: {
                observables.cmin1 = first.value
                if (sameContrastCheckbox.checked) {
                    observables.cmin2 = first.value
                }
            }
            second.onMoved: {
                observables.cmax1 = second.value
                if (sameContrastCheckbox.checked) {
                    observables.cmax2 = second.value
                }
            }
            Layout.row: 2
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
            Layout.row: 1
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
            from: 0
            to: 512
            second.value: 256
            first.onMoved: {
                observables.cmin2 = first.value
            }
            second.onMoved: {
                observables.cmax2 = second.value
            }
            Layout.row: 2
            Layout.column: 1
            visible: !sameContrastCheckbox.checked
        }
        JuliaCanvas {
            id: viewport3
            paintFunction: paint_cfunction3
            Layout.fillWidth: true
            Layout.preferredWidth: 300
            Layout.minimumWidth: 50
            Layout.minimumHeight: width
            Layout.maximumHeight: width
            Layout.row: 1
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
            Layout.row: 2
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
            Layout.row: 1
            Layout.column: 3
        }

        JuliaCanvas {
            id: viewport5
            paintFunction: paint_cfunction5
            Layout.fillWidth: true
            Layout.preferredWidth: 300
            Layout.minimumWidth: 50
            Layout.minimumHeight: width
            Layout.maximumHeight: width
            Layout.row: 1
            Layout.column: 4
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
            //ctx.save();
            ctx.clearRect( 0, 0, width, height);
            ctx.rect(0, 0, width, height);
            ctx.fillStyle = "#555555"
            ctx.fill()
            for(var trace_id=0; trace_id<2; trace_id += 1) {
                var y = observables["trace"+("RCS"[trace_id])]
                var tmin = Math.max(observables["tmin"], 1)
                var tmax = Math.min(y.length, observables["tmax"])
                var y_max = 1.5
                if(trace_id == 1) {
                    ctx.strokeStyle = observables["traceCol"]
                } else {
                    ctx.strokeStyle = "#AAAAAA"
                }
                ctx.beginPath()
                ctx.moveTo(0, (1-(y[tmin-1]+.4)/y_max)*height)
                for(var i=tmin; i<tmax; i+=1) {
                    ctx.lineTo((i-tmin)*width/(tmax-tmin), (1-((y[i]+.4))/y_max)*height)
                }
                ctx.stroke()
            }
            //Draw line at current time
            ctx.strokeStyle = Material.accent//"#749be8"
            ctx.fillStyle = Material.accent//"#749be8"
            var n_frames = observables["n_frames"]
            var tmin = observables["tmin"]
            var tmax = Math.min(n_frames, observables["tmax"])
            var frame_n = observables["frame_n"]
            var x = (frame_n - tmin)/(tmax - tmin) * width
            var arrowwidth = 8
            ctx.beginPath()
            ctx.moveTo(x, arrowwidth*2)
            ctx.lineTo(x-arrowwidth, arrowwidth*0.5)
            ctx.lineTo(x, 0)
            ctx.lineTo(x+arrowwidth, arrowwidth*0.5)
            ctx.lineTo(x, arrowwidth*2)
            ctx.lineTo(x, height)
            ctx.fill()
            ctx.stroke()

            //Draw time bar
            var barwidth_frames = 5
            var label = "250ms"
            if ((tmax - tmin) > 30*60*20) {
                barwidth_frames = 5*60*20
                label = "5min"
            } else if ((tmax - tmin) > 5*60*20) {
                barwidth_frames = 60*20
                label = "1min"
            } else if ((tmax - tmin) > 2*60*20) {
                barwidth_frames = 30*20
                label = "30s"
            } else if ((tmax - tmin) > 45*20) {
                barwidth_frames = 10*20
                label = "10s"
            } else if ((tmax - tmin) > 20*20) {
                barwidth_frames = 5*20
                label = "5s"
            } else if ((tmax - tmin) > 10*20) {
                barwidth_frames = 1*20
                label = "1s"
            }
            var barwidth = barwidth_frames / (tmax - tmin) * width
            ctx.strokeStyle = "#000000"
            ctx.fillStyle = "#000000"
            ctx.beginPath()
            ctx.moveTo(width-20, height-20)
            ctx.lineTo(width-20-barwidth, height-20)
            ctx.font = "14px serif"
            ctx.textAlign = "center"
            ctx.fillText(label, width -  20 - barwidth/2, height-5);
            ctx.stroke()
            //ctx.restore()
        }
        MouseArea {
            anchors.fill: parent
            property int lastX
            property int lastY
            onReleased: (mouse) => {
                var n_frames = parseInt(observables["n_frames"])
                var tmin = parseInt(observables["tmin"])
                var tmax = Math.min(n_frames, observables["tmax"])
                var rel_x = mouse.x/width
                var t = (tmin + rel_x*(tmax - tmin))/n_frames
                time_slider.value = t
            }
            onWheel: (mouse) => {
                Julia.zoomscrolltrace(mouse.x/width, mouse.y/height, mouse.angleDelta.y/8, observables)
            }
            onPressed: (mouse) => {
                lastX = mouse.x
                lastY = mouse.y
            }
            onPositionChanged: (mouse) => {
                Julia.pandragtrace((mouse.x - lastX)/width, (mouse.y - lastY)/height, observables)
                lastX = mouse.x
                lastY = mouse.y
            }
        }
        Keys.onPressed: {
            console.log(event.key)
        }
    }

    RowLayout {
        Slider {
            id: time_slider
            value: 0.0
            Layout.fillWidth: true
            Layout.fillHeight: false
            Layout.minimumWidth: 100
            onValueChanged: {
                observables.frame_n_float = value;
            }
        }
        Text {
            id: timeSliderLabel
            text: ""+observables.frame_n
            color: "#FFFFFF"
        }
        Button {
            id: stepBackButton
            Layout.preferredHeight: 50
            Layout.preferredWidth: height
            background: Canvas {
                Layout.fillWidth: true
                Layout.fillHeight: true
                contextType: "2d"
                antialiasing: true
                onPaint: {
                    var ctx = getContext( "2d" );
                    ctx.clearRect( 0, 0, width, height);
                    var rad = Math.min(width, height)/2
                    ctx.arc(width/2, height/2, rad, 0, 2*Math.PI)
                    ctx.fillStyle = "#555555"
                    ctx.fill()
                    ctx.beginPath()
                    ctx.moveTo(width/2-rad/2, height/2)
                    ctx.lineTo(width/2, height/2+rad/2.5)
                    ctx.lineTo(width/2, height/2-rad/2.5)
                    ctx.closePath()
                    ctx.rect(width*0.58, height/2-rad/2.5, rad*0.08, rad/1.25)
                    ctx.strokeStyle = Material.accent
                    ctx.fillStyle = Material.accent
                    ctx.fill()
                    ctx.stroke()
                }
            }
            onClicked: {
                observables.frame_n_float -= 1.0/observables.n_frames
            }
        }
        Button {
            id: stepForwardButton
            Layout.preferredHeight: 50
            Layout.preferredWidth: height
            background: Canvas {
                Layout.fillWidth: true
                Layout.fillHeight: true
                contextType: "2d"
                antialiasing: true
                onPaint: {
                    var ctx = getContext( "2d" );
                    ctx.clearRect( 0, 0, width, height);
                    var rad = Math.min(width, height)/2
                    ctx.arc(width/2, height/2, rad, 0, 2*Math.PI)
                    ctx.fillStyle = "#555555"
                    ctx.fill()
                    ctx.beginPath()
                    ctx.moveTo(width/2+rad/2, height/2)
                    ctx.lineTo(width/2, height/2+rad/2.5)
                    ctx.lineTo(width/2, height/2-rad/2.5)
                    ctx.closePath()
                    ctx.rect(width*0.4, height/2-rad/2.5, rad*0.08, rad/1.25)
                    ctx.strokeStyle = Material.accent
                    ctx.fillStyle = Material.accent
                    ctx.fill()
                    ctx.stroke()
                }
            }
            onClicked: {
                observables.frame_n_float += 1.0/observables.n_frames
            }
        }
        Button {
            id: playButton
            Layout.preferredHeight: 50
            Layout.preferredWidth: height
            background: Canvas {
                Layout.fillWidth: true
                Layout.fillHeight: true
                contextType: "2d"
                antialiasing: true
                onPaint: {
                    var ctx = getContext( "2d" );
                    ctx.clearRect( 0, 0, width, height);
                    var rad = Math.min(width, height)/2
                    ctx.arc(width/2, height/2, rad, 0, 2*Math.PI)
                    ctx.fillStyle = "#555555"
                    ctx.fill()
                    ctx.beginPath()
                    ctx.moveTo(width/2+rad/2, height/2)
                    ctx.lineTo(width/2-rad/4, height/2+rad/2.5)
                    ctx.lineTo(width/2-rad/4, height/2-rad/2.5)
                    ctx.closePath()
                    ctx.strokeStyle = Material.accent
                    ctx.fillStyle = Material.accent
                    ctx.fill()
                    ctx.stroke()
                }
            }
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
        else if (disp_id == 5) viewport5.update()
        else if (disp_id == 6) traceCanvas.requestPaint()
    }
  }

  FileDialog {
    id: openVideoDialog
    title: "Open video"
    folder: shortcuts.home
    selectMultiple: false
    selectExisting: true
    selectFolder: false
    onAccepted: {
        Julia.openvideo(openVideoDialog.fileUrl)
    }
    //onRejected: {
    //    console.log("Canceled")
    //}
    //Component.onCompleted: visible = true
  }

  FileDialog {
    id: saveResultDialog
    title: "Save results"
    folder: shortcuts.home
    selectMultiple: false
    selectExisting: false
    selectFolder: false
    onAccepted: {
        Julia.saveresult(saveResultDialog.fileUrl)
    }
    //onRejected: {
    //    console.log("Canceled")
    //}
    //Component.onCompleted: visible = true
  }

  FileDialog {
    id: openBehaviorDialog
    title: "Open behavior video"
    folder: shortcuts.home
    selectMultiple: false
    selectExisting: true
    selectFolder: false
    onAccepted: {
        Julia.openbehaviorvideo(openBehaviorDialog.fileUrl)
    }
    //onRejected: {
    //    console.log("Canceled")
    //}
    //Component.onCompleted: visible = true
  }

  FileDialog {
    id: openResultsDialog
    title: "Open results"
    folder: shortcuts.home
    selectMultiple: false
    selectExisting: true
    selectFolder: false
    onAccepted: {
        Julia.openresult(openResultsDialog.fileUrl)
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
    viewport5.update()
    timer.start();
  }
}