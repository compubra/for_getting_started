function slBusOut = Image(msgIn, slBusOut, varargin)
%#codegen
%   Copyright 2021-2022 The MathWorks, Inc.
    currentlength = length(slBusOut.header);
    for iter=1:currentlength
        slBusOut.header(iter) = bus_conv_fcns.ros2.msgToBus.std_msgs.Header(msgIn.header(iter),slBusOut(1).header(iter),varargin{:});
    end
    slBusOut.header = bus_conv_fcns.ros2.msgToBus.std_msgs.Header(msgIn.header,slBusOut(1).header,varargin{:});
    slBusOut.height = uint32(msgIn.height);
    slBusOut.width = uint32(msgIn.width);
    slBusOut.encoding_SL_Info.ReceivedLength = uint32(strlength(msgIn.encoding));
    currlen  = min(slBusOut.encoding_SL_Info.ReceivedLength, length(slBusOut.encoding));
    slBusOut.encoding_SL_Info.CurrentLength = uint32(currlen);
    slBusOut.encoding(1:currlen) = uint8(char(msgIn.encoding(1:currlen))).';
    slBusOut.is_bigendian = uint8(msgIn.is_bigendian);
    slBusOut.step = uint32(msgIn.step);
    maxlength = length(slBusOut.data);
    recvdlength = length(msgIn.data);
    currentlength = min(maxlength, recvdlength);
    if (max(recvdlength) > maxlength) && ...
            isequal(varargin{1}{1},ros.slros.internal.bus.VarLenArrayTruncationAction.EmitWarning)
        diag = MSLDiagnostic([], ...
                             message('ros:slros:busconvert:TruncatedArray', ...
                                     'data', msgIn.MessageType, maxlength, max(recvdlength), maxlength, varargin{2}));
        reportAsWarning(diag);
    end
    slBusOut.data_SL_Info.ReceivedLength = uint32(recvdlength);
    slBusOut.data_SL_Info.CurrentLength = uint32(currentlength);
    slBusOut.data = uint8(msgIn.data(1:slBusOut.data_SL_Info.CurrentLength));
    if recvdlength < maxlength
    slBusOut.data(recvdlength+1:maxlength) = 0;
    end
end
