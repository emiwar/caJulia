
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
                text: (100*observables.status_progress)+"%"
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
    RowLayout {
        Layout.preferredHeight: 50
        JuliaCanvas {
            id: viewport1
            paintFunction: paint_cfunction1
            Layout.fillWidth: true
            //Layout.fillHeight: true
            Layout.minimumWidth: 50
            Layout.minimumHeight: width
            Layout.maximumHeight: width
        }
        JuliaCanvas {
            id: viewport2
            paintFunction: paint_cfunction2
            Layout.fillWidth: true
            //Layout.fillHeight: true
            Layout.minimumWidth: 50
            Layout.minimumHeight: width
            Layout.maximumHeight: width
        }
        JuliaCanvas {
            id: viewport3
            paintFunction: paint_cfunction3
            Layout.fillWidth: true
            //Layout.fillHeight: true
            Layout.minimumWidth: 50
            Layout.minimumHeight: width
            Layout.maximumHeight: width
        }
        JuliaCanvas {
            id: viewport4
            paintFunction: paint_cfunction4
            Layout.fillWidth: true
            //Layout.fillHeight: true
            Layout.minimumWidth: 50
            Layout.minimumHeight: width
            Layout.maximumHeight: width
            MouseArea {
                anchors.fill: parent
                onClicked:(mouse)=>console.log(mouse.x)
            }
        }
    }
    Rectangle {
        Layout.fillWidth: true
        Layout.fillHeight: true
        color: "white"
    }
    RowLayout {
        Slider {
            id: time_slider
            value: 0.0
            //minimumValue: 0.0
            //maximumValue: 100.0
            Layout.fillWidth: true
            Layout.fillHeight: false
            Layout.minimumWidth: 100
            //Layout.minimumHeight: 100
            onValueChanged: {
                observables.fractional_time = value;
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
        }
        Button {
            id: stepForwardButton
            text: "🡢"
            Layout.preferredWidth: height
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