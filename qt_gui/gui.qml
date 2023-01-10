
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
  //Material.theme: Material.Dark

  menuBar: MenuBar {
        Menu {
            title: qsTr("File")
            Action { 
                text: qsTr("Open video") 
                onTriggered: fileDialog.visible = true;
            }
            Action { 
                text: qsTr("Quit") 
                onTriggered: Qt.quit();
            }
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
                observables.current_time = value;
                viewport1.update();
                viewport2.update();
            }
        }
        Button {
            id: stepBackButton
            text: "<"
            Layout.preferredWidth: height
        }
        Button {
            id: stepForwardButton
            text: ">"
            Layout.preferredWidth: height
        }
        Button {
            id: playButton
            text: "|>"
            Layout.preferredWidth: height
        }
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
}


